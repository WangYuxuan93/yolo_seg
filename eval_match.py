import os
import argparse
from difflib import SequenceMatcher
from shapely.geometry import box as shapely_box

def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB

    boxA_shapely = shapely_box(xa1, ya1, xa2, ya2)
    boxB_shapely = shapely_box(xb1, yb1, xb2, yb2)

    inter = boxA_shapely.intersection(boxB_shapely).area
    union = boxA_shapely.union(boxB_shapely).area

    return inter / union if union != 0 else 0

def load_boxes_from_txt(txt_path):
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            coords_str = parts[-4:]
            try:
                coords = list(map(int, coords_str))
            except ValueError:
                print(f"[WARNING] 非法坐标行已跳过: {line.strip()}")
                continue
            text = ' '.join(parts[:-4])
            boxes.append((text, coords))
    return boxes

import string

def evaluate(gold_dir, pred_dir, iou_thresh=0.8, match_min_sim=0.99, rm_punct=False):
    no_text_match_files = []
    box_text_mismatch_files = []
    no_text_match_files = []
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_text_matches = 0
    all_text_total = 0

    file_list = sorted(os.listdir(gold_dir))
    for fname in file_list:
        if not fname.endswith('.txt'):
            continue
        gold_path = os.path.join(gold_dir, fname)
        pred_path = os.path.join(pred_dir, fname)
        if not os.path.exists(pred_path):
            print(f"[!] Missing prediction for {fname}")
            continue

        gold_boxes = load_boxes_from_txt(gold_path)
        pred_boxes = load_boxes_from_txt(pred_path)

        matched_pred = set()
        matched_gold = set()
        text_match_count = 0
        any_text_matched = False
        text_mismatch_found = False
        mismatched_text_pairs = []

        for gi, (gtext, gbox) in enumerate(gold_boxes):
            best_iou = 0
            best_pi = -1
            for pi, (ptext, pbox) in enumerate(pred_boxes):
                if pi in matched_pred:
                    continue
                iou_score = iou(gbox, pbox)
                if iou_score >= iou_thresh and iou_score > best_iou:
                    best_iou = iou_score
                    best_pi = pi

            if best_pi >= 0:
                matched_gold.add(gi)
                matched_pred.add(best_pi)
                all_tp += 1
                # 比较文本
                ptext, _ = pred_boxes[best_pi]
                gt = gtext.replace(' ', '')
                pt = ptext.replace(' ', '')
                
                if rm_punct:
                    chinese_punct = "！？。；，、（）【】《》“”‘’"
                    all_punct = string.punctuation + chinese_punct
                    translator = str.maketrans('', '', all_punct)
                    gt = gt.translate(translator)
                    pt = pt.translate(translator)

                ratio = SequenceMatcher(None, gt, pt).ratio()
                if ratio > match_min_sim:
                    text_match_count += 1
                    any_text_matched = True
                else:
                    text_mismatch_found = True
                    mismatched_text_pairs.append((gtext, ptext))
            else:
                all_fn += 1

        all_fp += len(pred_boxes) - len(matched_pred)
        all_text_matches += text_match_count
        all_text_total += len(matched_gold)

        if not any_text_matched:
            no_text_match_files.append(fname)
        if text_mismatch_found:
            box_text_mismatch_files.append(fname)
            print(f"[✗] 文本不匹配示例（{fname}）：")
            for gtext, ptext in mismatched_text_pairs:
                print(f"  GT: {gtext} | Pred: {ptext}")

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    text_acc = all_text_matches / all_text_total if all_text_total > 0 else 0.0

    print("[Evaluation Results]")
    if box_text_mismatch_files:
        print("[!] 以下文件中有 box 匹配但文本错误：")
        for fname in box_text_mismatch_files:
            print(f" - {fname}")
    if no_text_match_files:
        print(f"[!] 以下文件没有匹配到任何文本：")
        for fname in no_text_match_files:
            print(f" - {fname}")
    print(f"Box Precision: {precision:.4f}")
    print(f"Box Recall:    {recall:.4f}")
    print(f"Box F1:        {f1:.4f}")
    print(f"Text Accuracy: {text_acc:.4f} (exact match > {match_min_sim:.2f} similarity)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_dir', type=str, required=True, help="路径：gold txt 文件夹")
    parser.add_argument('--pred_dir', type=str, required=True, help="路径：pred txt 文件夹")
    parser.add_argument('--iou_thresh', type=float, default=0.8, help="IOU 阈值")
    parser.add_argument('--match_min_sim', type=float, default=0.99, help="IOU 阈值")
    parser.add_argument('--rm_punct', action='store_true', help="匹配文本时是否去除所有标点")
    args = parser.parse_args()

    evaluate(args.gold_dir, args.pred_dir, iou_thresh=args.iou_thresh, match_min_sim=args.match_min_sim, rm_punct=args.rm_punct)
