#!/bin/bash
# run_ablation_via_evaluate_full.sh
# Uses the SAME evaluate_full.py that produced the confirmed 22.7% result.
# Just adds AdaFace-SR variant checkpoints via --adaface_checkpoint.
# This guarantees identical gallery selection, embedder, and eval protocol.
#
# Run on DGX:
#   cd /raid/home/dgxuser8/capstone1/version1
#   source ../capstone1/bin/activate
#   chmod +x run_ablation_via_evaluate_full.sh
#   CUDA_VISIBLE_DEVICES=0 ./run_ablation_via_evaluate_full.sh

set -e
BASE=/raid/home/dgxuser8/capstone1/version1
PALF_CKPT=$BASE/experiments/palfnet_v1/checkpoints/checkpoint_best.pth

echo "========================================================"
echo "Ablation: adaface_nogate"
echo "========================================================"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_full.py \
    --checkpoint $PALF_CKPT \
    --adaface_checkpoint $BASE/experiments/adaface_nogate/checkpoints/checkpoint_best.pth \
    --all_distances \
    --skip_sr_baselines \
    2>&1 | tee results/ablation_nogate.txt

echo ""
echo "========================================================"
echo "Ablation: adaface_nocomp"
echo "========================================================"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_full.py \
    --checkpoint $PALF_CKPT \
    --adaface_checkpoint $BASE/experiments/adaface_nocomp/checkpoints/checkpoint_best.pth \
    --all_distances \
    --skip_sr_baselines \
    2>&1 | tee results/ablation_nocomp.txt

echo ""
echo "========================================================"
echo "Ablation: adaface_noresenc"
echo "========================================================"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_full.py \
    --checkpoint $PALF_CKPT \
    --adaface_checkpoint $BASE/experiments/adaface_noresenc/checkpoints/checkpoint_best.pth \
    --all_distances \
    --skip_sr_baselines \
    2>&1 | tee results/ablation_noresenc.txt

echo ""
echo "========================================================"
echo "DONE. Results in results/ablation_*.txt"
echo "Grep for Rank-1 numbers:"
echo "  grep 'AdaFace-SR' results/ablation_*.txt"
echo "========================================================"
