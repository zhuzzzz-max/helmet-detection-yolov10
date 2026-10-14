# -*- coding: utf-8 -*-
"""阶段1-步骤2：把 SHWD 的 Pascal VOC 标注批量转换为 YOLO 格式。

VOC 标注：每个 xml 中包含图像尺寸和若干 <object>，框坐标为绝对像素
[xmin, ymin, xmax, ymax]（左上角 + 右下角，从 1 开始）。
YOLO 标注：每行 "class_id cx cy w h"，全部按图像宽高归一化到 [0,1]，
中心点 + 宽高表示。

输出：data/processed/labels_all/<图片名>.txt
同时生成 outputs/reports/conversion_report.json，记录转换统计与异常清单。

边界处理：
- 截断越界坐标到图像范围（并记录 warning 数量）；
- 跳过截断后面积为 0 的无效框；
- 忽略 difficult=1 的目标（SHWD 基本没有，保持通用）；
- 若 xml 缺少尺寸信息，用 PIL 读取真实图像尺寸；
- 不在类别映射中的 name 计入 unknown_classes，不写入标签。

用法：
    python scripts/02_voc_to_yolo.py
"""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter

from common import (LABELS_ALL_DIR, REPORTS_DIR, VOC_TO_YOLO_CLASS,
                    find_voc_dirs)

try:
    from PIL import Image
except ImportError:
    print("[错误] 需要 Pillow，请先执行：pip install Pillow")
    sys.exit(1)


def get_image_size(xml_root, img_path):
    """优先从 xml 读取尺寸，缺失或为 0 时用 PIL 读真实图片。"""
    size_node = xml_root.find("size")
    width = height = None
    if size_node is not None:
        w_node = size_node.find("width")
        h_node = size_node.find("height")
        if w_node is not None and h_node is not None:
            width = int(float(w_node.text))
            h_node = int(float(h_node.text))
    if not width or not height:
        with Image.open(img_path) as im:
            width, height = im.size
    return width, height


def voc_bbox_to_yolo(xmin, ymin, xmax, ymax, w, h):
    """绝对像素 VOC 框 -> 归一化 YOLO 框，并截断到图像边界。"""
    # 截断到 [0, 边长]
    xmin = max(0.0, min(xmin, w))
    xmax = max(0.0, min(xmax, w))
    ymin = max(0.0, min(ymin, h))
    ymax = max(0.0, min(ymax, h))

    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        return None

    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    nw = bw / w
    nh = bh / h
    # 归一化数值保险截断到 [0,1]
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(nw, 0.0), 1.0),
        min(max(nh, 0.0), 1.0),
    )


def main() -> int:
    voc_dirs = find_voc_dirs()
    if not voc_dirs:
        print("[错误] 未找到 VOC 数据集，请先运行 01_download_shwd.py 或手动放置数据。")
        return 1

    LABELS_ALL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    class_counter = Counter()
    n_images = 0
    n_xml = 0
    n_empty_labels = 0
    n_clamped = 0
    n_invalid = 0
    missing_images = []
    parse_errors = []
    unknown_classes = Counter()
    image_sources = {}  # stem -> 图片绝对路径，供划分步骤使用

    for voc_dir, ann_dir, img_dir in voc_dirs:
        print(f"[转换] {voc_dir.name} ...")
        xml_files = sorted(ann_dir.glob("*.xml"))
        for xml_path in xml_files:
            n_xml += 1
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError as e:
                parse_errors.append({"xml": str(xml_path), "error": str(e)})
                continue

            # 定位图片（文件名以 xml 中 filename 为准，缺失则回退到同名 + 常见扩展名）
            filename_node = root.find("filename")
            stem = xml_path.stem
            img_path = None
            if filename_node is not None and filename_node.text:
                candidate = img_dir / filename_node.text
                if candidate.exists():
                    img_path = candidate
            if img_path is None:
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    candidate = img_dir / (stem + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
            if img_path is None:
                missing_images.append(str(xml_path))
                continue

            try:
                w, h = get_image_size(root, img_path)
            except Exception as e:  # 图片损坏无法读取
                parse_errors.append({"xml": str(xml_path), "error": f"图像读取失败: {e}"})
                continue

            yolo_lines = []
            for obj in root.iter("object"):
                name_node = obj.find("name")
                if name_node is None or not name_node.text:
                    continue
                name = name_node.text.strip().lower()
                difficult_node = obj.find("difficult")
                difficult = int(difficult_node.text) if difficult_node is not None else 0
                if difficult == 1:
                    continue
                if name not in VOC_TO_YOLO_CLASS:
                    unknown_classes[name] += 1
                    continue

                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                try:
                    xmin = float(bnd.find("xmin").text)
                    ymin = float(bnd.find("ymin").text)
                    xmax = float(bnd.find("xmax").text)
                    ymax = float(bnd.find("ymax").text)
                except (AttributeError, TypeError, ValueError):
                    n_invalid += 1
                    continue

                if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
                    n_clamped += 1
                yolo_box = voc_bbox_to_yolo(xmin, ymin, xmax, ymax, w, h)
                if yolo_box is None:
                    n_invalid += 1
                    continue

                cls_id = VOC_TO_YOLO_CLASS[name]
                class_counter[cls_id] += 1
                yolo_lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in yolo_box))

            label_path = LABELS_ALL_DIR / (stem + ".txt")
            label_path.write_text("\n".join(yolo_lines), encoding="utf-8")
            if not yolo_lines:
                n_empty_labels += 1
            image_sources[stem] = str(img_path)
            n_images += 1

    # 保存“图片名 -> 原图路径”清单，供 03 划分使用，避免重复扫描
    (REPORTS_DIR / "image_sources.json").write_text(
        json.dumps(image_sources, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "xml_total": n_xml,
        "converted_images": n_images,
        "instances_per_class": {
            ("hat" if k == 0 else "head"): v for k, v in sorted(class_counter.items())
        },
        "empty_label_images": n_empty_labels,
        "out_of_range_boxes_clamped": n_clamped,
        "invalid_boxes_skipped": n_invalid,
        "unknown_classes": dict(unknown_classes),
        "missing_images": missing_images,
        "parse_errors": parse_errors,
    }
    (REPORTS_DIR / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    print(f"标注文件总数：{n_xml}")
    print(f"成功转换图片：{n_images}")
    print(f"类别实例统计：{dict(class_counter)}")
    print(f"空标签图片：{n_empty_labels}；越界截断框：{n_clamped}；无效框：{n_invalid}")
    if unknown_classes:
        print(f"[警告] 出现未映射类别：{dict(unknown_classes)}")
    if missing_images:
        print(f"[警告] {len(missing_images)} 个标注找不到对应图片，详见报告。")
    if parse_errors:
        print(f"[警告] {len(parse_errors)} 个文件解析失败，详见报告。")
    print(f"报告已保存：{REPORTS_DIR / 'conversion_report.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
