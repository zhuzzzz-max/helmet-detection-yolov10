# -*- coding: utf-8 -*-
"""阶段1一键流水线：下载 -> VOC 转 YOLO -> 划分 -> 筛查与可视化。

用法：
    python scripts/run_stage1.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_download_shwd.py",
    "02_voc_to_yolo.py",
    "03_split_dataset.py",
    "04_visualize_and_check.py",
]


def main() -> int:
    here = Path(__file__).resolve().parent
    for name in SCRIPTS:
        print("\n" + "#" * 70)
        print(f"# 运行 {name}")
        print("#" * 70)
        ret = subprocess.run([sys.executable, str(here / name)], shell=False)
        if ret.returncode != 0:
            print(f"[中断] {name} 失败，退出码 {ret.returncode}，请按提示排错后重跑。")
            return ret.returncode
    print("\n[完成] 阶段1全部结束。数据集位于 data/processed，报告位于 outputs/reports。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
