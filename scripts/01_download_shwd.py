# -*- coding: utf-8 -*-
"""阶段1-步骤1：下载 SHWD 公开安全帽数据集。

行为：
1. 若 data/raw/Safety-Helmet-Wearing-Dataset 已存在且包含 VOC 目录，则跳过；
2. 否则调用 git clone 拉取官方仓库（约数百 MB，耗时取决于网络）；
3. 若本机没有 git 或网络失败，给出手动下载指引，不做任何静默兜底。

用法：
    python scripts/01_download_shwd.py
"""
import shutil
import subprocess
import sys

from common import RAW_DIR, SHWD_LOCAL_DIR, SHWD_REPO_URL, find_voc_dirs


def already_downloaded() -> bool:
    """判断数据集是否已经就位：至少发现一个 VOC 目录且含 xml 文件。"""
    for voc_dir, ann_dir, img_dir in find_voc_dirs():
        n_xml = len(list(ann_dir.glob("*.xml")))
        n_img = sum(1 for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if n_xml > 0 and n_img > 0:
            print(f"[跳过] 已发现数据集：{voc_dir}（{n_xml} 个标注，{n_img} 张图片）")
            return True
    return False


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if already_downloaded():
        return 0

    if shutil.which("git") is None:
        print("[错误] 未检测到 git，请先安装 Git for Windows，或按下面方式手动下载。")
        _manual_hint()
        return 1

    print(f"[下载] git clone {SHWD_REPO_URL}")
    print(f"       目标目录：{SHWD_LOCAL_DIR}")
    cmd = ["git", "clone", "--depth", "1", SHWD_REPO_URL, str(SHWD_LOCAL_DIR)]
    try:
        ret = subprocess.run(cmd, shell=False)
    except OSError as e:
        print(f"[错误] 调用 git 失败：{e}")
        _manual_hint()
        return 1

    if ret.returncode != 0:
        print("[错误] git clone 失败，通常是网络问题（GitHub 访问不稳定）。")
        _manual_hint()
        return 1

    if not already_downloaded():
        print("[错误] 下载完成但未找到 VOC 格式目录，请检查仓库内容。")
        return 1

    print("[完成] SHWD 数据集下载成功。")
    return 0


def _manual_hint() -> None:
    print("-" * 70)
    print("手动下载方式：")
    print("1. 浏览器打开 https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset")
    print("2. 下载仓库 ZIP（绿色 Code 按钮 -> Download ZIP），或按 README 中的")
    print("   Google Drive / 百度网盘链接下载完整数据。")
    print(f"3. 解压后把 VOC2028 整个目录放到：{RAW_DIR}")
    print("   最终形如 data/raw/Safety-Helmet-Wearing-Dataset/VOC2028/")
    print("   （VOC2028 下应包含 Annotations、ImageSets、JPEGImages 三个目录）")
    print("4. 重新运行本脚本，检测到数据后会自动跳过下载。")
    print("-" * 70)


if __name__ == "__main__":
    sys.exit(main())
