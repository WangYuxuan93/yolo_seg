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

def main(pred_dir, gold_dir, output_json_path):
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

        # Per-image IoU
        iou = compute_union_iou(gold_polys, pred_polys)
        iou_dict[name] = round(iou, 6)

        # 累积到全局列表中
        all_pred_polys.extend(pred_polys)
        all_gold_polys.extend(gold_polys)

    # 按图像级 IoU 从高到低排序
    sorted_iou = dict(sorted(iou_dict.items(), key=lambda item: item[1], reverse=True))

    # 全局级 IoU 计算
    global_iou = compute_union_iou(all_gold_polys, all_pred_polys)
    sorted_iou["__global_iou__"] = round(global_iou, 6)

    # 写入 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_iou, f, indent=2)

    # 控制台输出
    if iou_dict:
        avg_iou = sum(iou_dict.values()) / len(iou_dict)
        print(f"\n📊 共评估 {len(iou_dict)} 张图")
        print(f"📈 图像级平均 IoU = {avg_iou:.4f}")
        print(f"🌐 全局级整体 IoU  = {global_iou:.4f}")
        print(f"✅ 结果已保存到 {output_json_path}")
    else:
        print("⚠️ 没有成功评估任何文件。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate overall IoU between predicted and gold polygons")
    parser.add_argument('--pred_dir', type=str, required=True, help="Path to prediction .txt files")
    parser.add_argument('--gold_dir', type=str, required=True, help="Path to gold labeled folders")
    parser.add_argument('--output_json', type=str, default='iou_results.json', help="Path to output JSON file")

    args = parser.parse_args()
    main(args.pred_dir, args.gold_dir, args.output_json)
