# -*- coding: utf-8 -*-
"""阶段2：测试集评估 + 检测结果可视化 + 指标汇总。

做了什么：
1. 加载训练得到的 best.pt，在 val 与 test 上分别评估，输出
   Precision、Recall、mAP@0.5、mAP@0.5:0.95；
2. 把关键指标与每类 AP 写入 outputs/reports/metrics_summary.json；
3. 从 test 集随机抽 N 张做推理，保存带检测框的可视化图。

用法：
    python scripts/06_val_predict.py
    python scripts/06_val_predict.py --weights outputs/runs/shwd_yolov10n/weights/best.pt --n 24
"""
import argparse
import json
import random
import sys
from pathlib import Path

from common import PROCESSED_DIR, REPORTS_DIR, RANDOM_SEED


def find_default_weights() -> Path:
    """未显式给权重时，自动找最近一次训练的 best.pt。"""
    runs = sorted((PROCESSED_DIR.parent.parent / "outputs" / "runs").glob("*"))
    candidates = [r / "weights" / "best.pt" for r in runs if (r / "weights" / "best.pt").exists()]
    if not candidates:
        return None
    return candidates[-1]


def eval_split(model, data_yaml: str, split: str):
    """在指定 split 上评估，返回指标字典。"""
    metrics = model.val(data=data_yaml, split=split, verbose=False, plots=True)
    box = metrics.box
    # 每类指标（按类别索引对齐）
    per_class = {}
    names = model.names
    for i, ap50 in enumerate(box.ap50):
        per_class[names.get(i, str(i))] = {
            "AP50": round(float(ap50), 4),
            "AP50_95": round(float(box.ap[i]), 4),
        }
    return {
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="测试集评估与结果可视化")
    parser.add_argument("--weights", default=None, help="best.pt 路径")
    parser.add_argument("--data", default=str(REPORTS_DIR.parent / "runtime_shwd.yaml"))
    parser.add_argument("--n", type=int, default=24, help="test 集可视化张数")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else find_default_weights()
    if weights is None or not weights.exists():
        print("[错误] 未找到 best.pt，请先运行 05_train.py，或用 --weights 指定。")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics，请先执行：pip install ultralytics")
        return 1

    model = YOLO(str(weights))

    summary = {"weights": str(weights), "metrics": {}}
    for split in ("val", "test"):
        print(f"[评估] split = {split}")
        summary["metrics"][split] = eval_split(model, args.data, split)
        m = summary["metrics"][split]
        print(f"  P={m['precision']}  R={m['recall']}  "
              f"mAP@0.5={m['mAP50']}  mAP@0.5:0.95={m['mAP50_95']}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / "metrics_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # 测试集可视化
    test_img_dir = PROCESSED_DIR / "images" / "test"
    images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    rng = random.Random(args.seed)
    rng.shuffle(images)
    picked = images[:args.n]
    if picked:
        vis_out = REPORTS_DIR / "detections"
        model.predict(
            source=[str(p) for p in picked],
            save=True,
            project=str(vis_out),
            name="test",
            conf=0.25,
            iou=0.6,
            imgsz=640,
            exist_ok=True,
        )
        print(f"[可视化] 检测结果图已保存到 {vis_out / 'test'}")

    print("=" * 60)
    print(f"[完成] 指标汇总：{out_json}")
    print("可直接把 mAP@0.5、mAP@0.5:0.95 与训练曲线 results.png 用于结果整理。")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
