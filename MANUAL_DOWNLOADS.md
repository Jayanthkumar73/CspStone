# Manual Download Instructions

If automated downloads fail (e.g., Google Drive quota), download manually:

## Pretrained Models

### AdaFace IR-101 (WebFace4M) — Face Recognition Backbone
- URL: https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT
- Save to: `pretrained/recognition/adaface_ir101_webface4m.ckpt`

### 6DRepNet — Pose Estimation
- URL: https://drive.google.com/file/d/1BpHMhxbP2gZbeYBmTMVOSMhMlQc37RUj
- Save to: `pretrained/pose/6DRepNet_300W_LP_AFLW2000.pth`

### GFPGAN v1.4 — Face Super-Resolution Baseline
- URL: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
- Save to: `pretrained/sr/GFPGANv1.4.pth`

### Real-ESRGAN x4plus — General Super-Resolution Baseline
- URL: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
- Save to: `pretrained/sr/RealESRGAN_x4plus.pth`

## Datasets

### TinyFace — https://qmul-tinyface.github.io/
- Unzip into: `data/tinyface/`

### QMUL-SurvFace — https://qmul-survface.github.io/
- Unzip into: `data/survface/`

### SCface — http://www.scface.org/ (requires application)
- Unzip into: `data/scface/`
- Then run: `python scripts/organize_scface.py`

### CASIA-WebFace (for training) — https://www.kaggle.com/datasets/debargo/casia-webface-cleaned
- Unzip into: `data/training/casia-webface/`
