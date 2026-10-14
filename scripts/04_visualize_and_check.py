# -*- coding: utf-8 -*-
"""阶段1-步骤4：脏数据筛查 + 样本可视化。

脏数据筛查项（逐图检查，结果写入 outputs/reports/dataset_report.txt 与
issues.csv）：
1. 损坏图片（PIL 无法打开/解码）；
2. 有图无标签 / 有标签无图；
3. 标签列数错误、类别 id 越界、坐标超出 [0,1]；
4. 零面积框、极小框（宽或高 < 2 像素）；
5. 图片重复（按文件大小 + MD5 去重）；
6. 统计类别分布、框尺寸分布（按 COCO 口径：面积开方 / 图像尺寸）、
   分辨率分布。

样本可视化：从 train/val/test 各随机抽若干张，按 YOLO 标签画框，
拼成网格图保存到 outputs/reamples/（带标签，便于人工核对）。

用法：
    python scripts/04_visualize_and_check.py
    python scripts/04_visualize_and_check.py --samples 16
"""
import argparse
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import (BOX_COLORS, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR,
                    YOLO_CLASS_NAMES)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("[错误] 需要 Pillow，请先执行：pip install Pillow")
    sys.exit(1)


def md5_of(path: Path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_split(split: str, issues: list, stats: dict):
    """检查单个 split 的全部图片与标签。"""
    img_dir = PROCESSED_DIR / "images" / split
    lbl_dir = PROCESSED_DIR / "labels" / split
    images = sorted([p for p in img_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])

    for img_path in images:
        stem = img_path.stem
        lbl_path = lbl_dir / (stem + ".txt")
        stats["per_split"][split] += 1
        stats["size_buckets"][img_path.stat().st_size] = img_path

        # 1) 图片可读性
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                w, h = im.size
                im.load()
        except Exception as e:
            issues.append((split, stem, "corrupt_image", str(e)[:100]))
            continue

        stats["resolution"][f"{w}x{h}"] += 1

        # 2) 标签存在性
        if not lbl_path.exists():
            issues.append((split, stem, "missing_label", ""))
            continue

        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        if not lines:
            stats["empty_label"] += 1
            issues.append((split, stem, "empty_label", ""))
            continue

        for ln_no, ln in enumerate(lines, 1):
            parts = ln.split()
            if len(parts) != 5:
                issues.append((split, stem, f"bad_column_count@line{ln_no}", ln))
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                issues.append((split, stem, f"bad_number@line{ln_no}", ln))
                continue

            if cls_id not in YOLO_CLASS_NAMES:
                issues.append((split, stem, f"class_id_out_of_range@line{ln_no}", str(cls_id)))
                continue
            stats["class_count"][YOLO_CLASS_NAMES[cls_id]] += 1

            if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                issues.append((split, stem, f"coord_out_of_range@line{ln_no}", ln))
                continue
            if bw <= 0 or bh <= 0:
                issues.append((split, stem, f"zero_area@line{ln_no}", ln))
                continue

            # 框尺度：等效边长占比（近似 COCO 小目标口径）
            side_ratio = (bw * bh) ** 0.5
            if side_ratio < 0.03:
                stats["box_size"]["small"] += 1
            elif side_ratio < 0.1:
                stats["box_size"]["medium"] += 1
            else:
                stats["box_size"]["large"] += 1

            if bw * w < 2 or bh * h < 2:
                issues.append((split, stem, f"tiny_box<2px@line{ln_no}", ln))

    # 有标签无图
    if lbl_dir.exists():
        img_stems = {p.stem for p in images}
        for lbl_path in lbl_dir.glob("*.txt"):
            if lbl_path.stem not in img_stems:
                issues.append((split, lbl_path.stem, "orphan_label", ""))


def find_duplicates(stats):
    """同大小文件再比 MD5，返回 md5 -> [路径] 的重复组（只保留 >1 的）。"""
    dup_groups = []
    by_size = stats["size_buckets"]
    size_to_files = defaultdict(list)
    for size, p in by_size.items():
        size_to_files[size].append(p)
    for size, files in size_to_files.items():
        if len(files) < 2:
            continue
        md5_map = defaultdict(list)
        for p in files:
            md5_map[md5_of(p)].append(p)
        for digest, group in md5_map.items():
            if len(group) > 1:
                dup_groups.append(group)
    return dup_groups


def draw_sample_grid(split: str, n_samples: int, seed: int, out_path: Path):
    """抽取 n_samples 张图画框，拼 4 列网格。"""
    img_dir = PROCESSED_DIR / "images" / split
    lbl_dir = PROCESSED_DIR / "labels" / split
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(images)
    images = images[:n_samples]
    if not images:
        return False

    cell_w, cell_h = 480, 320
    cols = 4
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for idx, img_path in enumerate(images):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            scale = min(cell_w / w0, cell_h / h0)
            new_w, new_h = int(w0 * scale), int(h0 * scale)
            tile = im.resize((new_w, new_h))
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h
        canvas.paste(tile, (x0, y0))
        td = ImageDraw.Draw(canvas)
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for ln in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = x0 + (cx - bw / 2) * new_w
                y1 = y0 + (cy - bh / 2) * new_h
                x2 = x0 + (cx + bw / 2) * new_w
                y2 = y0 + (cy + bh / 2) * new_h
                color = BOX_COLORS.get(cls_id, (255, 0, 255))
                td.rectangle([x1, y1, x2, y2], outline=color, width=2)
                td.text((x1 + 2, max(0, y1 - 16)),
                        YOLO_CLASS_NAMES.get(cls_id, "?"),
                        fill=color, font=font)
        draw.text((x0 + 4, y0 + cell_h - 18), img_path.name[:30],
                  fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="脏数据筛查与可视化")
    parser.add_argument("--samples", type=int, default=16, help="每个 split 抽样张数")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if not PROCESSED_DIR.exists():
        print("[错误] 未找到 processed 数据集，请先运行步骤 2、3。")
        return 1

    issues = []
    stats = {
        "per_split": Counter(),
        "class_count": Counter(),
        "box_size": Counter(),
        "resolution": Counter(),
        "empty_label": 0,
        "size_buckets": {},
    }

    for split in ("train", "val", "test"):
        check_split(split, issues, stats)

    dup_groups = find_duplicates(stats)
    for group in dup_groups:
        for p in group:
            issues.append(("duplicate", p.stem, "duplicate_image",
                           " | ".join(str(x) for x in group)))

    # issues.csv
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    issues_csv = REPORTS_DIR / "issues.csv"
    with open(issues_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "stem", "issue_type", "detail"])
        writer.writerows(issues)

    # 文本报告
    lines = []
    lines.append("SHWD 数据质量筛查报告")
    lines.append("=" * 60)
    lines.append(f"各 split 样本数：{dict(stats['per_split'])}")
    lines.append(f"类别实例数：{dict(stats['class_count'])}")
    lines.append(f"框尺度分布（小/中/大，等效边长占比 3%/10% 分界）："
                 f"{dict(stats['box_size'])}")
    lines.append(f"空标签图片数：{stats['empty_label']}")
    lines.append(f"重复图片组数：{len(dup_groups)}")
    lines.append(f"问题记录总数：{len(issues)}")
    issue_type_counter = Counter(t for _, _, t, _ in issues)
    lines.append("问题类型统计：")
    for t, n in issue_type_counter.most_common():
        lines.append(f"  - {t}: {n}")
    lines.append("")
    lines.append("分辨率 Top10：")
    for res, n in stats["resolution"].most_common(10):
        lines.append(f"  - {res}: {n}")
    (REPORTS_DIR / "dataset_report.txt").write_text(
        "\n".join(lines), encoding="utf-8")

    # 可视化
    vis_dir = REPORTS_DIR / "samples"
    for split in ("train", "val", "test"):
        out = vis_dir / f"samples_{split}.jpg"
        ok = draw_sample_grid(split, args.samples, args.seed, out)
        if ok:
            print(f"[可视化] 已保存 {out}")

    print("\n".join(lines))
    print(f"\n问题明细：{issues_csv}")
    print(f"文字报告：{REPORTS_DIR / 'dataset_report.txt'}")
    print("=" * 60)
    print("说明：empty_label / tiny_box 等仅为提示，SHWD 标注质量稳定，")
    print("      若无 corrupt_image 与大量越界记录即可进入训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
