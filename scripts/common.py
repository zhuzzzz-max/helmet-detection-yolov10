# -*- coding: utf-8 -*-
"""项目公共路径与常量配置。

所有脚本统一从这里取路径，保证在任意工作目录下运行结果一致。
目录结构：
helmet_yolov10/
├── data/
│   ├── raw/                 # SHWD 原始数据集（git clone 产物）
│   └── processed/           # 转换后的 YOLO 数据集
│       ├── images/{train,val,test}
│       ├── labels/{train,val,test}
│       └── labels_all/      # 全量转换标签（划分前）
├── outputs/                 # 训练输出、数据筛查报告
├── configs/
└── scripts/
"""
from pathlib import Path

# 项目根目录（scripts 的上一级）
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS_ALL_DIR = PROCESSED_DIR / "labels_all"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CONFIGS_DIR = ROOT / "configs"

# SHWD 官方仓库地址
SHWD_REPO_URL = "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset.git"
# clone 后的本地目录名
SHWD_LOCAL_DIR = RAW_DIR / "Safety-Helmet-Wearing-Dataset"

# 类别映射：SHWD 的 VOC 标注中 name 为 "hat"（戴安全帽）与 "person"（未戴安全帽的裸头）。
# YOLO 训练时把 "person" 重命名为语义更准确的 "head"，对应类别 id = 1。
VOC_TO_YOLO_CLASS = {
    "hat": 0,
    "person": 1,
}
YOLO_CLASS_NAMES = {
    0: "hat",
    1: "head",
}

# 数据集划分比例（train / val / test）与随机种子
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 可视化框颜色（BGR）
BOX_COLORS = {
    0: (0, 200, 0),    # hat 绿色
    1: (0, 0, 255),    # head 红色
}


def find_voc_dirs(raw_root: Path = RAW_DIR):
    """自动发现 raw 目录下所有 Pascal VOC 风格的数据目录。

    判定标准：目录下同时存在 Annotations 与 JPEGImages 子目录。
    SHWD 官方数据对应顶层目录名为 VOC2028，这样写也兼容其他镜像命名。
    返回 [(voc_dir, ann_dir, img_dir), ...]
    """
    results = []
    if not raw_root.exists():
        return results
    for ann_dir in raw_root.rglob("Annotations"):
        if not ann_dir.is_dir():
            continue
        voc_dir = ann_dir.parent
        img_dir = voc_dir / "JPEGImages"
        if img_dir.is_dir():
            results.append((voc_dir, ann_dir, img_dir))
    # 去重并按路径排序，保证可复现
    seen = set()
    unique = []
    for item in results:
        key = str(item[0])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return sorted(unique, key=lambda x: str(x[0]))
