#!/usr/bin/env python3
"""
download_models.py — Download all pretrained models needed for PALF-Net.

Usage:
    python version1/scripts/download_models.py
    python version1/scripts/download_models.py --model adaface
    python version1/scripts/download_models.py --model pose
    python version1/scripts/download_models.py --model gfpgan
    python version1/scripts/download_models.py --model realesrgan
    python version1/scripts/download_models.py --model insightface
"""

import os, sys, subprocess, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, desc=""):
    print(f"  → {desc}...")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            print(f"    ✅ Done")
            return True
        print(f"    ❌ {r.stderr[:200]}")
    except Exception as e:
        print(f"    ❌ {e}")
    return False


def dl_adaface():
    print("\n── AdaFace IR-101 WebFace4M ──")
    p = f"{ROOT}/pretrained/recognition/adaface_ir101_webface4m.ckpt"
    if os.path.isfile(p) and os.path.getsize(p) > 1e8:
        print(f"  ✅ Exists ({os.path.getsize(p)/1e6:.0f} MB)")
        return True
    return run(f'gdown "https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT" -O "{p}"',
               "Google Drive via gdown")


def dl_insightface():
    print("\n── InsightFace buffalo_l (ArcFace R100) ──")
    d = f"{ROOT}/pretrained/recognition"
    if os.path.isdir(f"{d}/models/buffalo_l"):
        print("  ✅ Exists")
        return True
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', root=d,
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  ✅ Downloaded")
        return True
    except Exception as e:
        print(f"  ❌ {e}")
        return False


def dl_pose():
    print("\n── 6DRepNet Pose ──")
    p = f"{ROOT}/pretrained/pose/6DRepNet_300W_LP_AFLW2000.pth"
    if os.path.isfile(p) and os.path.getsize(p) > 1e7:
        print(f"  ✅ Exists ({os.path.getsize(p)/1e6:.0f} MB)")
        return True
    return run(f'gdown "https://drive.google.com/uc?id=1BpHMhxbP2gZbeYBmTMVOSMhMlQc37RUj" -O "{p}"',
               "Google Drive via gdown")


def dl_gfpgan():
    print("\n── GFPGAN v1.4 ──")
    p = f"{ROOT}/pretrained/sr/GFPGANv1.4.pth"
    if os.path.isfile(p) and os.path.getsize(p) > 1e8:
        print(f"  ✅ Exists ({os.path.getsize(p)/1e6:.0f} MB)")
        return True
    return run(f'wget -q --show-progress '
               f'"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" '
               f'-O "{p}"', "GitHub releases via wget")


def dl_realesrgan():
    print("\n── Real-ESRGAN x4plus ──")
    p = f"{ROOT}/pretrained/sr/RealESRGAN_x4plus.pth"
    if os.path.isfile(p) and os.path.getsize(p) > 1e7:
        print(f"  ✅ Exists ({os.path.getsize(p)/1e6:.0f} MB)")
        return True
    return run(f'wget -q --show-progress '
               f'"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" '
               f'-O "{p}"', "GitHub releases via wget")


MODELS = {
    "adaface": dl_adaface, "insightface": dl_insightface,
    "pose": dl_pose, "gfpgan": dl_gfpgan, "realesrgan": dl_realesrgan,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["all"] + list(MODELS.keys()))
    args = parser.parse_args()

    print("=" * 50)
    print("  PALF-Net Model Downloader")
    print("=" * 50)

    todo = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    res = {k: fn() for k, fn in todo.items()}

    print("\n" + "=" * 50)
    for k, v in res.items():
        print(f"  {k:15s}: {'✅' if v else '❌'}")
    failed = sum(1 for v in res.values() if not v)
    if failed:
        print(f"\n  {failed} failed. See MANUAL_DOWNLOADS.md")
    else:
        print(f"\n  🎉 All models ready!")
