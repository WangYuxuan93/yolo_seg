import os
import glob
import json
import argparse
from shapely.geometry import Polygon
from shapely.ops import unary_union

def load_polygons_from_txt(txt_path, has_label=False):
    polys = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if has_label:
                parts = line.strip().split()
            else:
                parts = line.strip().split(',')
            coords = list(map(float, parts[-8:]))
            pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            poly = Polygon(pts)
            if poly.is_valid:
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

def compute_box_level_metrics(pred_dir, gold_dir, iou_threshold=0.9):
    correct = 0
    total_pred = 0
    total_gold = 0

    for pred_txt in glob.glob(os.path.join(pred_dir, '*.txt')):
        name = os.path.splitext(os.path.basename(pred_txt))[0]
        gold_txt = os.path.join(gold_dir, name, 'gold_item', 'item_box.txt')

        if not os.path.exists(gold_txt):
            continue

        pred_polys = load_polygons_from_txt(pred_txt)
        gold_polys = load_polygons_from_txt(gold_txt, has_label=True)

        total_pred += len(pred_polys)
        total_gold += len(gold_polys)

        matched_gold = set()

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

    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1, correct, total_pred, total_gold

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

    # 图像级排序
    sorted_iou = dict(sorted(iou_dict.items(), key=lambda item: item[1], reverse=True))

    # 全局级 IoU
    global_iou = compute_union_iou(all_gold_polys, all_pred_polys)
    sorted_iou["__global_iou__"] = round(global_iou, 6)

    # Box-level metrics
    box_p, box_r, box_f1, correct, total_pred, total_gold = compute_box_level_metrics(
        pred_dir, gold_dir, iou_threshold
    )
    sorted_iou[f"__box_precision@{iou_threshold}__"] = round(box_p, 6)
    sorted_iou[f"__box_recall@{iou_threshold}__"] = round(box_r, 6)
    sorted_iou[f"__box_f1@{iou_threshold}__"] = round(box_f1, 6)

    # 输出 JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_iou, f, indent=2)

    # 控制台输出
    if iou_dict:
        avg_iou = sum(iou_dict.values()) / len(iou_dict)
        print(f"\n📊 共评估 {len(iou_dict)} 张图")
        print(f"📈 图像级平均 IoU = {avg_iou:.4f}")
        print(f"🌐 全局级整体 IoU  = {global_iou:.4f}")
        print(f"🎯 Box Precision@{iou_threshold} = {box_p:.4f} ({correct}/{total_pred})")
        print(f"🔁 Box Recall@{iou_threshold}    = {box_r:.4f} ({correct}/{total_gold})")
        print(f"⭐ Box F1@{iou_threshold}        = {box_f1:.4f}")
        print(f"✅ 结果已保存到 {output_json_path}")
    else:
        print("⚠️ 没有成功评估任何文件。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate polygon-level and box-level IoU metrics")
    parser.add_argument('--pred_dir', type=str, required=True, help="Path to prediction .txt files")
    parser.add_argument('--gold_dir', type=str, required=True, help="Path to gold labeled folders")
    parser.add_argument('--output_json', type=str, default='iou_results.json', help="Path to output JSON file")
    parser.add_argument('--iou_threshold', type=float, default=0.9, help="IoU threshold for box-level precision/recall/F1")

    args = parser.parse_args()
    main(args.pred_dir, args.gold_dir, args.output_json, args.iou_threshold)
