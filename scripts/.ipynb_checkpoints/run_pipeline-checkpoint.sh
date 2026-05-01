#!/bin/bash
# ============================================================================
# run_pipeline.sh — Full AdaFace-SR pipeline: setup → train → evaluate
#
# Run this on your DGX A100. It does everything end-to-end.
#
# Prerequisites:
#   - CUDA GPU available
#   - Python 3.11 with PyTorch installed
#   - pip install insightface onnxruntime-gpu gfpgan realesrgan basicsr \
#               sixdrepnet lpips opencv-python-headless matplotlib
#   - SCface dataset at data/scface/organized/ with mugshot/ and surveillance/ dirs
#   - Training data at data/training/aligned_112/ (CASIA-WebFace or similar)
#
# Usage:
#   chmod +x scripts/run_pipeline.sh
#   nohup bash scripts/run_pipeline.sh > pipeline_log.txt 2>&1 &
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "=============================================="
echo "  AdaFace-SR Full Pipeline"
echo "  Started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="

# ── STEP 1: Download pretrained models ──
echo ""
echo "━━━ STEP 1/4: Setup pretrained models ━━━"
bash scripts/setup_pretrained.sh

# ── STEP 2: Download ArcFace PyTorch weights ──
echo ""
echo "━━━ STEP 2/4: Download ArcFace PyTorch weights ━━━"
python scripts/download_arcface_pytorch.py iresnet50

# ── STEP 3: Train AdaFace-SR v3 (with all fixes) ──
echo ""
echo "━━━ STEP 3/4: Train AdaFace-SR v3 ━━━"
echo "  Configuration:"
echo "    - Optimizer: AdamW (weight_decay=1e-4)"
echo "    - gate_weight=0.5, comparative_weight=2.0 (stable, no collapse)"
echo "    - LR range: 14-36px (focused on surveillance zone)"
echo "    - Epochs: 50, Batch: 8"
echo ""

python scripts/train_adaface_sr.py \
    --data_dir data/training/aligned_112 \
    --pose_cache data/training/poses_cache.json \
    --max_samples 50000 \
    --epochs 50 \
    --batch_size 8 \
    --num_workers 4 \
    --exp_name adaface_sr_v3_fixed

echo ""
echo "  ✅ Training complete"

# ── STEP 4: Evaluate ──
echo ""
echo "━━━ STEP 4/4: Evaluate on SCface (all distances) ━━━"

# Find the best checkpoint
CKPT="experiments/adaface_sr_v3_fixed/checkpoints/checkpoint_best.pth"
if [ ! -f "$CKPT" ]; then
    CKPT="experiments/adaface_sr_v3_fixed/checkpoints/checkpoint_last.pth"
fi

echo "  Using checkpoint: $CKPT"

# Check if PALF-Net checkpoint exists (optional baseline)
PALF_CKPT=""
for p in experiments/palfnet_v1/checkpoints/checkpoint_best.pth \
         experiments/palfnet_*/checkpoints/checkpoint_best.pth; do
    if [ -f "$p" ]; then
        PALF_CKPT="$p"
        break
    fi
done

if [ -n "$PALF_CKPT" ]; then
    echo "  PALF-Net checkpoint: $PALF_CKPT"
    python scripts/evaluate_full.py \
        --checkpoint "$PALF_CKPT" \
        --adaface_checkpoint "$CKPT" \
        --all_distances \
        --scface_dir data/scface
else
    echo "  ⚠️ No PALF-Net checkpoint found, evaluating AdaFace-SR only"
    # Run eval with AdaFace-SR as the main model
    # Create a dummy checkpoint arg (eval script requires it, but will skip if not found)
    python scripts/evaluate_full.py \
        --checkpoint "$CKPT" \
        --adaface_checkpoint "$CKPT" \
        --all_distances \
        --scface_dir data/scface
fi

echo ""
echo "=============================================="
echo "  Pipeline Complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
echo "    Report:  results/evaluation/evaluation_report.txt"
echo "    Plots:   results/evaluation/plots/"
echo "    JSON:    results/evaluation/evaluation_summary.json"
echo "    Samples: results/evaluation/visual_samples/"
echo ""
echo "  KEY NUMBER TO CHECK:"
echo "    Look for 'AdaFace-SR' at d3 16px in the report."
echo "    If it beats 22.7% (bicubic), the paper narrative flips!"
echo "=============================================="