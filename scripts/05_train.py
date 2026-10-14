# -*- coding: utf-8 -*-
"""阶段2：YOLOv10 基线训练（2 分类：hat / head）。

做了什么：
1. 读取 configs/shwd.yaml，把数据路径改写为本机绝对路径，生成
   outputs/runtime_shwd.yaml（ultralytics 对相对路径的解析依赖运行目录，
   显式绝对化最稳妥）；
2. 加载 YOLOv10n 预训练权重（首次运行自动下载，约 5MB）；
3. 按设定超参数训练，默认 100 epochs、imgsz=640、batch=16，
   适配 12GB 显存（RTX 3060/4070 级别）；
4. 训练结束自动在 val 上评估，并导出 best.pt 与训练曲线图。

用法：
    python scripts/05_train.py                         # 默认 yolov10n
    python scripts/05_train.py --model yolov10s.pt --epochs 100
    python scripts/05_train.py --device 0 --batch 16
"""
import argparse
import sys
from pathlib import Path

from common import CONFIGS_DIR, OUTPUTS_DIR, PROCESSED_DIR


def build_runtime_yaml() -> Path:
    """生成带绝对路径的运行时 data yaml。"""
    runtime_yaml = OUTPUTS_DIR / "runtime_shwd.yaml"
    content = (
        f"path: {PROCESSED_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "nc: 2\n"
        "names:\n"
        "  0: hat\n"
        "  1: head\n"
    )
    runtime_yaml.write_text(content, encoding="utf-8")
    return runtime_yaml


def check_dataset_ready() -> bool:
    """训练前最小检查：三个 split 的图片与标签目录都非空。"""
    ok = True
    for split in ("train", "val", "test"):
        img_dir = PROCESSED_DIR / "images" / split
        n = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        if n == 0:
            print(f"[错误] {split} 图片为空，请先完成阶段1。")
            ok = False
        else:
            print(f"[检查] {split}: {n} 张图片")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv10 安全帽检测基线训练")
    parser.add_argument("--model", default="yolov10n.pt",
                        choices=["yolov10n.pt", "yolov10s.pt", "yolov10m.pt"],
                        help="模型规模，默认 yolov10n（12G 显存推荐 n/s）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size；12G 显存 yolov10n 可用 16，yolov10s 建议 8")
    parser.add_argument("--device", default="0", help="GPU 编号，如 0；CPU 用 cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(OUTPUTS_DIR / "runs"))
    parser.add_argument("--name", default="shwd_yolov10n")
    args = parser.parse_args()

    if not check_dataset_ready():
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics，请先执行：pip install ultralytics")
        return 1

    data_yaml = build_runtime_yaml()
    print(f"[配置] 数据集 yaml：{data_yaml}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        # 常用稳定训练策略：余弦学习率 + 预热 + 轻微 mosaic 增强
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # 工地场景增强：HSV、翻转、马赛克；关闭过强的上下翻转
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=10,      # 最后 10 轮关 mosaic，稳定收敛
        patience=20,          # 20 轮无提升提前停
        save=True,
        plots=True,           # 自动输出 results.png、confusion_matrix.png
        val=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print("=" * 60)
    print(f"[完成] 训练输出目录：{save_dir}")
    print("关键文件：")
    print(f"  - 最优权重：{save_dir / 'weights' / 'best.pt'}")
    print(f"  - 训练曲线：{save_dir / 'results.png'}")
    print(f"  - 指标明细：{save_dir / 'results.csv'}")
    print("  - 混淆矩阵：confusion_matrix.png")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
