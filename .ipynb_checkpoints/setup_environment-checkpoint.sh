#!/bin/bash
###############################################################################
# PALF-Net Phase 0: Complete Environment Setup (DGX - No Conda)
#
# This script:
#   1. Installs all Python dependencies via pip
#   2. Downloads all pretrained models via wget/pip
#   3. Prints status of everything
#
# Usage:
#   cd /raid/home/dgxuser8/capstone1/version1
#   chmod +x setup_environment.sh
#   bash setup_environment.sh
###############################################################################

set -e

# Auto-detect ROOT as the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"

echo "=============================================="
echo "  PALF-Net Setup (DGX - No Conda)"
echo "  Root: $ROOT"
echo "=============================================="

# ─────────────────────────────────────────────
# STEP 1: Install Python Dependencies
# ─────────────────────────────────────────────
echo ""
echo "[1/4] Installing Python dependencies..."

pip install --upgrade pip

# Core ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    echo "  PyTorch may already be installed, continuing..."

pip install \
    numpy scipy scikit-learn matplotlib seaborn tqdm tensorboard pandas \
    opencv-python-headless Pillow scikit-image einops timm lpips \
    insightface onnxruntime-gpu \
    gfpgan realesrgan basicsr facexlib \
    sixdrepnet gdown pyyaml

echo "  ✅ Python dependencies installed"

# ─────────────────────────────────────────────
# STEP 2: Create directory structure
# ─────────────────────────────────────────────
echo ""
echo "[2/4] Ensuring directory structure..."

mkdir -p "$ROOT/configs"
mkdir -p "$ROOT/scripts"
mkdir -p "$ROOT/results"
mkdir -p "$ROOT/experiments"
mkdir -p "$ROOT/src/models"
mkdir -p "$ROOT/src/data"
mkdir -p "$ROOT/src/eval"
mkdir -p "$ROOT/src/utils"
mkdir -p "$ROOT/pretrained/recognition"
mkdir -p "$ROOT/pretrained/pose"
mkdir -p "$ROOT/pretrained/sr"
mkdir -p "$ROOT/data/tinyface"
mkdir -p "$ROOT/data/survface"
mkdir -p "$ROOT/data/scface/surveillance"
mkdir -p "$ROOT/data/scface/mugshot"
mkdir -p "$ROOT/data/training/aligned_112"
mkdir -p "$ROOT/results/phase0/visualizations"
mkdir -p "$ROOT/results/phase0/sr_samples"

# Python package init files
touch "$ROOT"/src/__init__.py
touch "$ROOT"/src/models/__init__.py
touch "$ROOT"/src/data/__init__.py
touch "$ROOT"/src/eval/__init__.py
touch "$ROOT"/src/utils/__init__.py

echo "  ✅ Directories ready"

# ─────────────────────────────────────────────
# STEP 3: Download Pretrained Models
# ─────────────────────────────────────────────
echo ""
echo "[3/4] Downloading pretrained models..."

# --- GFPGAN v1.4 ---
GFPGAN_PATH="$ROOT/pretrained/sr/GFPGANv1.4.pth"
if [ ! -f "$GFPGAN_PATH" ]; then
    echo "  Downloading GFPGAN v1.4..."
    wget -q --show-progress \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
        -O "$GFPGAN_PATH" && echo "  ✅ GFPGAN downloaded" || echo "  ❌ GFPGAN failed"
else
    echo "  ✅ GFPGAN already exists"
fi

# --- Real-ESRGAN x4plus ---
REALESRGAN_PATH="$ROOT/pretrained/sr/RealESRGAN_x4plus.pth"
if [ ! -f "$REALESRGAN_PATH" ]; then
    echo "  Downloading Real-ESRGAN..."
    wget -q --show-progress \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        -O "$REALESRGAN_PATH" && echo "  ✅ Real-ESRGAN downloaded" || echo "  ❌ Real-ESRGAN failed"
else
    echo "  ✅ Real-ESRGAN already exists"
fi

# --- 6DRepNet (via gdown from Google Drive) ---
POSE_PATH="$ROOT/pretrained/pose/6DRepNet_300W_LP_AFLW2000.pth"
if [ ! -f "$POSE_PATH" ]; then
    echo "  Downloading 6DRepNet pose model..."
    gdown "https://drive.google.com/uc?id=1BpHMhxbP2gZbeYBmTMVOSMhMlQc37RUj" \
        -O "$POSE_PATH" 2>/dev/null && echo "  ✅ 6DRepNet downloaded" || \
    echo "  ❌ 6DRepNet auto-download failed. Manual: https://drive.google.com/file/d/1BpHMhxbP2gZbeYBmTMVOSMhMlQc37RUj"
else
    echo "  ✅ 6DRepNet already exists"
fi

# --- AdaFace IR-101 (via gdown from Google Drive) ---
ADAFACE_PATH="$ROOT/pretrained/recognition/adaface_ir101_webface4m.ckpt"
if [ ! -f "$ADAFACE_PATH" ]; then
    echo "  Downloading AdaFace IR-101 WebFace4M..."
    gdown "https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT" \
        -O "$ADAFACE_PATH" 2>/dev/null && echo "  ✅ AdaFace downloaded" || \
    echo "  ❌ AdaFace auto-download failed. Manual: https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT"
else
    echo "  ✅ AdaFace already exists"
fi

# --- InsightFace buffalo_l (ArcFace R100) ---
INSIGHT_DIR="$ROOT/pretrained/recognition"
echo "  Downloading InsightFace buffalo_l (ArcFace R100)..."
python3 -c "
import os, sys
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', root='$INSIGHT_DIR', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640,640))
    print('  ✅ InsightFace buffalo_l downloaded')
except Exception as e:
    print(f'  ⚠️  InsightFace download issue: {e}')
    print('  Will retry when running verify_setup.py')
" 2>/dev/null || echo "  ⚠️  InsightFace setup will complete on first use"

# ─────────────────────────────────────────────
# STEP 4: Summary
# ─────────────────────────────────────────────
echo ""
echo "[4/4] Checking final state..."
echo ""
echo "  Pretrained models:"
for f in "$ROOT"/pretrained/recognition/*.ckpt "$ROOT"/pretrained/recognition/*.safetensors \
         "$ROOT"/pretrained/pose/*.pth "$ROOT"/pretrained/sr/*.pth; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "    ✅ $(basename $f) ($SIZE)"
    fi
done

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "    1. python $ROOT/scripts/verify_setup.py"
echo "    2. Place SCface data in: $ROOT/data/scface/"
echo "       Then run: python $ROOT/scripts/organize_scface.py"
echo "    3. python $ROOT/scripts/phase0_pose_feasibility.py"
echo "=============================================="
