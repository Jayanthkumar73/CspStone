# PALF-Net: Pose-Aware Low-Resolution Face Recognition

## Quick Start (DGX)

```bash
# 1. Copy version1/ to your DGX
scp -r version1/ dgxuser8@dgx:/raid/home/dgxuser8/capstone1/version1/

# 2. Install deps + download models (one command)
cd /raid/home/dgxuser8/capstone1/version1
chmod +x setup_environment.sh
bash setup_environment.sh

# 3. Verify everything
python scripts/verify_setup.py

# 4. Place SCface (unzip your download)
#    Put contents in: data/scface/
python scripts/organize_scface.py

# 5. Run Phase 0 (GO / NO-GO test)
python scripts/phase0_pose_feasibility.py

# With your SCface data:
python scripts/phase0_pose_feasibility.py --image_dir data/scface
```

## Execution Order

| Step | Command | What it does |
|------|---------|-------------|
| 0a | `bash setup_environment.sh` | Install deps, download models |
| 0b | `python scripts/verify_setup.py` | Check everything works |
| 0c | `python scripts/organize_scface.py` | Organize SCface data |
| **0d** | **`python scripts/phase0_pose_feasibility.py`** | **GO/NO-GO: Can we estimate pose from LR faces?** |
| 1 | `python scripts/phase1_baseline_eval.py` | Baseline FR + pose-stratified analysis |
| 2 | `python scripts/train_palfnet.py` | Train PALF-Net |
| 3 | `python scripts/evaluate.py` | Full evaluation |

## Project Structure

```
version1/
├── setup_environment.sh        ← RUN FIRST
├── MANUAL_DOWNLOADS.md         ← If auto-download fails
├── configs/
│   ├── train.yaml
│   └── eval.yaml
├── scripts/
│   ├── verify_setup.py         ← Step 0b
│   ├── download_models.py      ← Retry failed downloads
│   ├── organize_scface.py      ← Step 0c
│   ├── phase0_pose_feasibility.py  ← Step 0d (CRITICAL)
│   ├── phase1_baseline_eval.py     ← Step 1
│   ├── train_palfnet.py            ← Step 2
│   └── evaluate.py                 ← Step 3
├── src/
│   ├── models/
│   │   ├── pose_film.py        ← FiLM conditioning (THE NOVELTY)
│   │   ├── identity_loss.py    ← ArcFace identity loss
│   │   ├── sr_backbone.py      ← SR network (Phase 2)
│   │   └── palfnet.py          ← Full pipeline (Phase 2)
│   ├── data/
│   ├── eval/
│   └── utils/
│       ├── pose_estimator.py
│       ├── face_detector.py
│       └── metrics.py
├── pretrained/                 ← Auto-downloaded
├── data/                       ← Datasets go here
├── experiments/                ← Logs
└── results/                    ← Output tables & plots
```
