# -*- coding: utf-8 -*-
"""阶段1-步骤3：按 8:1:1 自动划分 train/val/test，并组装 YOLO 目录。

输出目录布局（ultralytics 官方约定）：
data/processed/
├── images/{train,val,test}/<图片名>
├── labels/{train,val,test}/<同名>.txt
├── train.txt / val.txt / test.txt   （每张图的绝对路径，供核对）

图片默认采用硬链接（hard link）进入 processed，避免重复占用约 1GB 磁盘；
硬链接失败时自动回退到复制。标签文件同理。

用法：
    python scripts/03_split_dataset.py
    python scripts/03_split_dataset.py --ratios 0.8 0.1 0.1 --seed 42
"""
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from common import (LABELS_ALL_DIR, PROCESSED_DIR, RANDOM_SEED,
                    REPORTS_DIR, SPLIT_RATIOS)


def link_or_copy(src: Path, dst: Path):
    """优先硬链接，跨卷/失败则复制。dst 父目录需已存在。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build_split(stems, ratios, seed):
    """按比例随机划分，返回 {'train': [...], 'val': [...], 'test': [...]}。"""
    rng = random.Random(seed)
    stems = sorted(stems)
    rng.shuffle(stems)
    n = len(stems)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    # 余数全部归 test，保证三者之和等于总数
    return {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="划分 SHWD 数据集")
    parser.add_argument("--ratios", type=float, nargs=3, default=list(SPLIT_RATIOS),
                        help="train/val/test 比例，默认 0.8 0.1 0.1")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    args = parser.parse_args()

    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        print(f"[错误] 比例之和必须为 1，当前为 {sum(ratios)}")
        return 1

    sources_path = REPORTS_DIR / "image_sources.json"
    if not sources_path.exists():
        print("[错误] 未找到 image_sources.json，请先运行 02_voc_to_yolo.py。")
        return 1
    image_sources = json.loads(sources_path.read_text(encoding="utf-8"))

    # 仅保留“图片存在且标签存在”的样本
    valid_stems = []
    missing = []
    for stem, src in image_sources.items():
        if Path(src).exists() and (LABELS_ALL_DIR / (stem + ".txt")).exists():
            valid_stems.append(stem)
        else:
            missing.append(stem)
    if missing:
        print(f"[警告] {len(missing)} 个样本缺少图片或标签，已排除。")

    if not valid_stems:
        print("[错误] 没有可用样本。")
        return 1

    splits = build_split(valid_stems, ratios, args.seed)

    link_stats = {"hardlink": 0, "copy": 0}
    for split, stems in splits.items():
        print(f"[组装] {split}: {len(stems)} 个样本")
        list_file = PROCESSED_DIR / f"{split}.txt"
        written_paths = []
        for stem in stems:
            src_img = Path(image_sources[stem])
            src_lbl = LABELS_ALL_DIR / (stem + ".txt")
            dst_img = PROCESSED_DIR / "images" / split / src_img.name
            dst_lbl = PROCESSED_DIR / "labels" / split / (stem + ".txt")
            link_stats[link_or_copy(src_img, dst_img)] += 1
            link_stats[link_or_copy(src_lbl, dst_lbl)] += 1
            written_paths.append(str(dst_img.resolve()))
        list_file.write_text("\n".join(written_paths), encoding="utf-8")

    split_report = {
        "seed": args.seed,
        "ratios": dict(zip(["train", "val", "test"], ratios)),
        "counts": {k: len(v) for k, v in splits.items()},
        "excluded_missing": missing,
        "link_stats": link_stats,
    }
    (REPORTS_DIR / "split_report.json").write_text(
        json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    for k, v in splits.items():
        print(f"{k}: {len(v)}")
    print(f"文件组织方式：{link_stats}")
    print(f"报告已保存：{REPORTS_DIR / 'split_report.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
