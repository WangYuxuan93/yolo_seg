import os
import glob
import json
import argparse
from collections import Counter
from shapely.geometry import Polygon
from shapely.ops import unary_union

def load_polygons_from_txt(txt_path, has_label=False, return_label=False):
    polys = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if has_label:
                parts = line.strip().split()
                label = parts[-9] if len(parts) >= 9 else None
            else:
                parts = line.strip().split(',')
                label = None
            coords = list(map(float, parts[-8:]))
            pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 0:
                if return_label:
                    polys.append((poly, label))
                else:
                    polys.append(poly)
    return polys

def compute_union_iou(gold_polys, pred_polys):
    gold_union = unary_union(gold_polys)
    pred_union = unary_union(pred_polys)
    if not gold_union.is_valid or not pred_union.is_valid:
        return 0.0
    inter = gold_union.intersection(pred_union).area
    union = gold_union.union(pred_union).area
    return inter / union if union != 0 else 0.0

def compute_image_prf(pred_dir, gold_dir, iou_threshold=0.9):
    prf_dict = {}
    precisions, recalls, f1s = [], [], []

    for pred_txt in glob.glob(os.path.join(pred_dir, '*.txt')):
        name = os.path.splitext(os.path.basename(pred_txt))[0]
        gold_txt = os.path.join(gold_dir, name, 'gold_item', 'item_box.txt')

        if not os.path.exists(gold_txt):
            continue

        pred_polys = load_polygons_from_txt(pred_txt)
        gold_polys = load_polygons_from_txt(gold_txt, has_label=True)

        matched_gold = set()
        correct = 0
        for pred in pred_polys:
            max_iou = 0
            max_idx = -1
            for idx, gold in enumerate(gold_polys):
                if idx in matched_gold:
                    continue
                iou = compute_union_iou([gold], [pred])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx
            if max_iou >= iou_threshold:
                correct += 1
                matched_gold.add(max_idx)

        total_pred = len(pred_polys)
        total_gold = len(gold_polys)
        precision = correct / total_pred if total_pred > 0 else 0.0
        recall = correct / total_gold if total_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        prf_dict[name] = {
            'precision': round(precision, 6),
            'recall': round(recall, 6),
            'f1': round(f1, 6),
            'correct': correct,
            'total_pred': total_pred,
            'total_gold': total_gold
        }

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    prf_dict["__avg_precision__"] = round(avg_precision, 6)
    prf_dict["__avg_recall__"] = round(avg_recall, 6)
    prf_dict["__avg_f1__"] = round(avg_f1, 6)

    return dict(sorted(prf_dict.items(), key=lambda x: x[1]['f1'] if isinstance(x[1], dict) else -1, reverse=True))

def compute_per_class_recall(pred_dir, gold_dir, iou_threshold=0.9):
    label_total = Counter()
    label_matched = Counter()

    for pred_txt in glob.glob(os.path.join(pred_dir, '*.txt')):
        name = os.path.splitext(os.path.basename(pred_txt))[0]
        gold_txt = os.path.join(gold_dir, name, 'gold_item', 'item_box.txt')

        if not os.path.exists(gold_txt):
            continue

        pred_polys = load_polygons_from_txt(pred_txt)
        gold_labeled = load_polygons_from_txt(gold_txt, has_label=True, return_label=True)

        matched_gold = set()

        for pred in pred_polys:
            max_iou = 0
            max_idx = -1
            for idx, (gold_poly, gold_label) in enumerate(gold_labeled):
                if idx in matched_gold or gold_label is None:
                    continue
                iou = compute_union_iou([gold_poly], [pred])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = idx
            if max_iou >= iou_threshold:
                matched_gold.add(max_idx)

        for idx, (gold_poly, gold_label) in enumerate(gold_labeled):
            if gold_label is None:
                continue
            label_total[gold_label] += 1
            if idx in matched_gold:
                label_matched[gold_label] += 1

    label_recall = {
        label: round(label_matched[label] / label_total[label], 6)
        for label in label_total
    }
    return label_recall

def count_label_distribution(gold_dir):
    label_counter = Counter()
    for gold_txt in glob.glob(os.path.join(gold_dir, '*', 'gold_item', 'item_box.txt')):
        labeled = load_polygons_from_txt(gold_txt, has_label=True, return_label=True)
        label_counter.update(label for _, label in labeled if label is not None)
    return label_counter

def main(pred_dir, gold_dir, output_json_path, iou_threshold):
    iou_dict = {}
    all_pred_polys = []
    all_gold_polys = []

    for pred_txt in glob.glob(os.path.join(pred_dir, '*.txt')):
        name = os.path.splitext(os.path.basename(pred_txt))[0]
        gold_txt = os.path.join(gold_dir, name, 'gold_item', 'item_box.txt')

        if not os.path.exists(gold_txt):
            print(f"[Skip] Missing gold for {name}")
            continue

        pred_polys = load_polygons_from_txt(pred_txt)
        gold_polys = load_polygons_from_txt(gold_txt, has_label=True)

        if not pred_polys or not gold_polys:
            print(f"[Warning] Empty polygons in {name}")
            continue

        iou = compute_union_iou(gold_polys, pred_polys)
        iou_dict[name] = round(iou, 6)

        all_pred_polys.extend(pred_polys)
        all_gold_polys.extend(gold_polys)

    sorted_iou = dict(sorted(iou_dict.items(), key=lambda item: item[1], reverse=True))
    global_iou = compute_union_iou(all_gold_polys, all_pred_polys)
    sorted_iou["__global_iou__"] = round(global_iou, 6)

    label_counts = count_label_distribution(gold_dir)
    sorted_iou["__label_distribution__"] = dict(label_counts)

    # 加入各类别召回率
    per_class_recall = compute_per_class_recall(pred_dir, gold_dir, iou_threshold)
    sorted_iou["__per_class_recall__"] = per_class_recall

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_iou, f, indent=2)

    # 单图 PRF 保存
    prf_per_image = compute_image_prf(pred_dir, gold_dir, iou_threshold)
    with open(output_json_path.replace(".json", "_prf.json"), 'w', encoding='utf-8') as f:
        json.dump(prf_per_image, f, indent=2)

    if iou_dict:
        avg_iou = sum(iou_dict.values()) / len(iou_dict)
        avg_precision = prf_per_image.get("__avg_precision__", 0.0)
        avg_recall = prf_per_image.get("__avg_recall__", 0.0)
        avg_f1 = prf_per_image.get("__avg_f1__", 0.0)
        print(f"\n📊 共评估 {len(iou_dict)} 张图")
        print(f"📈 图像级平均 IoU = {avg_iou:.4f}")
        print(f"🌐 全局级整体 IoU  = {global_iou:.4f}")
        print(f"📊 各类别数量分布：{dict(label_counts)}")
        print(f"📊 各类别召回率：{per_class_recall}")
        print(f"📊 平均 Precision = {avg_precision:.4f}")
        print(f"📊 平均 Recall    = {avg_recall:.4f}")
        print(f"📊 平均 F1        = {avg_f1:.4f}")
        print(f"📂 单图 PRF 排序结果保存至 {output_json_path.replace('.json', '_prf.json')}")
    else:
        print("⚠️ 没有成功评估任何文件。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate polygon-level and box-level IoU metrics")
    parser.add_argument('--pred_dir', type=str, required=True, help="Path to prediction .txt files")
    parser.add_argument('--gold_dir', type=str, required=True, help="Path to gold labeled folders")
    parser.add_argument('--output_json', type=str, default='iou_results.json', help="Path to output JSON file")
    parser.add_argument('--iou_threshold', type=float, default=0.8, help="IoU threshold for box-level precision/recall/F1")
    args = parser.parse_args()
    main(args.pred_dir, args.gold_dir, args.output_json, args.iou_threshold)
