#!/bin/bash
# ============================================================================
# setup_pretrained.sh — Download ALL pretrained models needed for AdaFace-SR
#
# Run this ONCE on your DGX before training or evaluation.
#
# Usage:
#   chmod +x scripts/setup_pretrained.sh
#   bash scripts/setup_pretrained.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  AdaFace-SR: Downloading Pretrained Models"
echo "  Root: $ROOT"
echo "=============================================="

# --- 1. ArcFace iResNet-50 (for DIFFERENTIABLE identity loss during training) ---
ARCFACE_DIR="$ROOT/pretrained/arcface_pytorch"
mkdir -p "$ARCFACE_DIR"

if [ ! -f "$ARCFACE_DIR/iresnet50.pth" ]; then
    echo ""
    echo "[1/5] Downloading ArcFace iResNet-50..."
    wget -q --show-progress -O "$ARCFACE_DIR/iresnet50.pth" \
        "https://sota.nizhib.ai/pytorch-insightface/iresnet50-7f187506.pth"
    echo "  ✅ ArcFace iResNet-50 saved"
else
    echo "[1/5] ArcFace iResNet-50 already exists ✅"
fi

# --- 2. InsightFace buffalo_l (for EVALUATION embedding extraction) ---
# InsightFace auto-downloads, but we can pre-create the directory
RECOG_DIR="$ROOT/pretrained/recognition"
mkdir -p "$RECOG_DIR/models"
echo "[2/5] InsightFace buffalo_l will auto-download on first use ✅"

# --- 3. GFPGAN v1.4 (SR baseline) ---
SR_DIR="$ROOT/pretrained/sr"
mkdir -p "$SR_DIR"

if [ ! -f "$SR_DIR/GFPGANv1.4.pth" ]; then
    echo ""
    echo "[3/5] Downloading GFPGAN v1.4..."
    wget -q --show-progress -O "$SR_DIR/GFPGANv1.4.pth" \
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    echo "  ✅ GFPGAN v1.4 saved"
else
    echo "[3/5] GFPGAN v1.4 already exists ✅"
fi

# --- 4. Real-ESRGAN x4plus (SR baseline) ---
if [ ! -f "$SR_DIR/RealESRGAN_x4plus.pth" ]; then
    echo ""
    echo "[4/5] Downloading Real-ESRGAN x4plus..."
    wget -q --show-progress -O "$SR_DIR/RealESRGAN_x4plus.pth" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    echo "  ✅ Real-ESRGAN x4plus saved"
else
    echo "[4/5] Real-ESRGAN x4plus already exists ✅"
fi

# --- 5. SixDRepNet (pose estimation — used by PALF-Net baseline) ---
POSE_DIR="$ROOT/pretrained/pose"
mkdir -p "$POSE_DIR"
echo "[5/5] SixDRepNet auto-downloads via pip package ✅"

# --- Verify ---
echo ""
echo "=============================================="
echo "  Verification"
echo "=============================================="

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo "  ✅ $2: $SIZE"
    else
        echo "  ❌ $2: MISSING"
    fi
}

check_file "$ARCFACE_DIR/iresnet50.pth" "ArcFace iResNet-50"
check_file "$SR_DIR/GFPGANv1.4.pth" "GFPGAN v1.4"
check_file "$SR_DIR/RealESRGAN_x4plus.pth" "Real-ESRGAN x4plus"

echo ""
echo "=============================================="
echo "  Setup complete! You can now train and evaluate."
echo "=============================================="