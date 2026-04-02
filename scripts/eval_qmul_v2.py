"""
QMUL-SurvFace Verification Evaluation — v2
Uses AdaFace IR-101 (WebFace4M) consistently as FR backbone.
Directly comparable to official leaderboard (CentreFace/SphereFace/FaceNet).

Metrics: AUC, TAR@FAR=30%, 10%, 1%, 0.1%

Usage:
    python scripts/eval_qmul_v2.py \
        --qmul_dir      data/QMUL-SurvFace \
        --fr_backbone   pretrained/recognition/adaface_ir101_webface4m.ckpt \
        --adaface_v2    experiments/adaface_sr_v2/checkpoints/checkpoint_best.pth \
        --adaface_qmul  experiments/adaface_qmul_v1/checkpoints/checkpoint_best.pth \
        > eval_qmul_v2_results.txt 2>&1
"""

import os, sys, time, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, '/raid/home/dgxuser8/capstone1/version1')

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='QMUL-SurvFace Verification — AdaFace IR-101')
parser.add_argument('--qmul_dir',     required=True)
parser.add_argument('--fr_backbone',  required=True,
                    help='AdaFace IR-101 checkpoint (.ckpt)')
parser.add_argument('--adaface_v2',   required=True)
parser.add_argument('--adaface_qmul', required=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n{'='*70}")
print(f"  QMUL-SurvFace Verification — AdaFace IR-101 Backbone")
print(f"  Comparable to official leaderboard")
print(f"  Device: {device}")
print(f"{'='*70}\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
VER_DIR = os.path.join(args.qmul_dir,
                       'Face_Verification_Test_Set', 'verification_images')
POS_MAT = os.path.join(args.qmul_dir,
                       'Face_Verification_Test_Set', 'positive_pairs_names.mat')
NEG_MAT = os.path.join(args.qmul_dir,
                       'Face_Verification_Test_Set', 'negative_pairs_names.mat')

# ── Load pairs ────────────────────────────────────────────────────────────────
print("[1/5] Loading verification pairs...")

def parse_pairs_mat(mat_path):
    mat = loadmat(mat_path)
    for k, v in mat.items():
        if k.startswith('_'):
            continue
        arr = np.array(v)
        if arr.ndim == 2 and arr.shape[1] == 2:
            pairs = []
            for row in arr:
                f1 = str(row[0][0]) if hasattr(row[0], '__iter__') else str(row[0])
                f2 = str(row[1][0]) if hasattr(row[1], '__iter__') else str(row[1])
                pairs.append((f1.strip(), f2.strip()))
            return pairs
    raise ValueError(f"Cannot parse {mat_path}")

pos_pairs = parse_pairs_mat(POS_MAT)
neg_pairs = parse_pairs_mat(NEG_MAT)
print(f"  Positive pairs: {len(pos_pairs)}")
print(f"  Negative pairs: {len(neg_pairs)}")

# ── Load AdaFace IR-101 backbone ──────────────────────────────────────────────
print("\n[2/5] Loading AdaFace IR-101 (WebFace4M) backbone...")

from adaface_ir101_loader import load_adaface_ir101, extract_embedding as _extract_emb

fr_model = load_adaface_ir101(args.fr_backbone, device)

def extract_embedding(img_bgr):
    return _extract_emb(fr_model, img_bgr, device)

# Quick score gap diagnostic
print("\n  Score gap diagnostic (100 pairs)...")
from scipy.io import loadmat as _lm
_pos = parse_pairs_mat(POS_MAT)
_neg = parse_pairs_mat(NEG_MAT)
_ps, _ns = [], []
for f1, f2 in _pos[:100]:
    e1 = extract_embedding(cv2.imread(os.path.join(VER_DIR, f1)))
    e2 = extract_embedding(cv2.imread(os.path.join(VER_DIR, f2)))
    if e1 is not None and e2 is not None:
        _ps.append(float(np.dot(e1, e2)))
for f1, f2 in _neg[:100]:
    e1 = extract_embedding(cv2.imread(os.path.join(VER_DIR, f1)))
    e2 = extract_embedding(cv2.imread(os.path.join(VER_DIR, f2)))
    if e1 is not None and e2 is not None:
        _ns.append(float(np.dot(e1, e2)))
print(f"  Genuine:  mean={np.mean(_ps):.4f}  std={np.std(_ps):.4f}")
print(f"  Impostor: mean={np.mean(_ns):.4f}  std={np.std(_ns):.4f}")
print(f"  Gap:      {np.mean(_ps)-np.mean(_ns):.4f}")

# ── Load SR models ────────────────────────────────────────────────────────────
print("\n[3/5] Loading SR models...")
from src.models.adaface_sr import AdaFaceSR

def load_sr(path, name):
    model = AdaFaceSR().to(device).eval()
    try:
        ckpt  = torch.load(path, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        print(f"  ✅ {name} loaded (epoch={ckpt.get('epoch','?')})")
        return model
    except Exception as e:
        print(f"  ❌ {name} failed: {e}")
        return None

adaface_v2   = load_sr(args.adaface_v2,   'AdaFace-SR v2 (synthetic)')
adaface_qmul = load_sr(args.adaface_qmul, 'AdaFace-SR QMUL (real LR)')

def enhance(img_bgr, sr_model):
    bic = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
    if sr_model is None:
        return bic
    rgb = cv2.cvtColor(bic, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    h, w = img_bgr.shape[:2]
    res  = torch.tensor([min(h, w)], dtype=torch.long).to(device)
    with torch.no_grad():
        try:    out = sr_model(t, res)
        except: out = sr_model(t)
        # Handle tuple output (some models return (output, gate) or similar)
        if isinstance(out, tuple):
            out = out[0]
        out = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# ── Cache embeddings ──────────────────────────────────────────────────────────
print("\n[4/5] Extracting embeddings for all verification images...")

all_images = sorted(set(
    f for pair in pos_pairs + neg_pairs for f in pair
))
print(f"  Unique images: {len(all_images)}")

methods = [
    ('Bicubic',         None),
    ('AdaFace-SR v2',   adaface_v2),
    ('AdaFace-SR QMUL', adaface_qmul),
]

method_embeddings = {}

for method_name, sr_model in methods:
    print(f"\n  Processing: {method_name}")
    cache  = {}
    failed = 0
    t0     = time.time()

    for i, fname in enumerate(all_images):
        img = cv2.imread(os.path.join(VER_DIR, fname))
        if img is None:
            failed += 1
            cache[fname] = None
            continue
        cache[fname] = extract_embedding(enhance(img, sr_model))
        if (i + 1) % 1000 == 0:
            print(f"    [{i+1}/{len(all_images)}]  {time.time()-t0:.0f}s")

    method_embeddings[method_name] = cache
    print(f"  Done — failed: {failed}  time: {time.time()-t0:.0f}s")

# ── Compute metrics ───────────────────────────────────────────────────────────
print("\n[5/5] Computing verification metrics...")

FAR_TARGETS = [0.30, 0.10, 0.01, 0.001]

def tar_at_far(labels, scores, far_target):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, far_target, side='right') - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx]) * 100

results = {}

for method_name in [m[0] for m in methods]:
    embs   = method_embeddings[method_name]
    scores, labels = [], []

    for f1, f2 in pos_pairs:
        e1, e2 = embs.get(f1), embs.get(f2)
        if e1 is not None and e2 is not None:
            scores.append(float(np.dot(e1, e2)))
            labels.append(1)

    for f1, f2 in neg_pairs:
        e1, e2 = embs.get(f1), embs.get(f2)
        if e1 is not None and e2 is not None:
            scores.append(float(np.dot(e1, e2)))
            labels.append(0)

    labels = np.array(labels)
    scores = np.array(scores)
    auc    = roc_auc_score(labels, scores) * 100
    tars   = [tar_at_far(labels, scores, f) for f in FAR_TARGETS]

    results[method_name] = {
        'auc': auc, 'tar30': tars[0], 'tar10': tars[1],
        'tar1': tars[2], 'tar01': tars[3], 'n': len(scores)
    }
    print(f"  {method_name}: AUC={auc:.1f}%  TAR@1%={tars[2]:.1f}%  "
          f"TAR@0.1%={tars[3]:.1f}%")

# ── Final report ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  QMUL-SurvFace VERIFICATION — AdaFace IR-101 Backbone")
print(f"  ({len(pos_pairs)} genuine + {len(neg_pairs)} impostor pairs)")
print(f"{'='*70}")
print(f"  {'Method':<24} {'AUC':>7} {'@30%':>7} {'@10%':>7} "
      f"{'@1%':>7} {'@0.1%':>7}")
print(f"  {'-'*65}")

for name, r in results.items():
    print(f"  {name:<24} {r['auc']:>6.1f}% {r['tar30']:>6.1f}% "
          f"{r['tar10']:>6.1f}% {r['tar1']:>6.1f}% {r['tar01']:>6.1f}%")

# Published baselines
baselines = {
    'CentreFace (ECCV16)': (94.8, 27.3, 13.8, 3.1),
    'SphereFace (CVPR17)': (83.9, 21.3,  8.3, 1.0),
    'FaceNet    (CVPR15)': (93.5, 12.7,  4.3, 1.0),
}
print(f"\n  ── Published SOTA (official leaderboard) ──")
print(f"  {'Method':<24} {'AUC':>7} {'@30%':>7} {'@10%':>7} {'@1%':>7}")
for name, (auc, t30, t10, t1) in baselines.items():
    print(f"  {name:<24} {auc:>6.1f}% {t30:>6.1f}% {t10:>6.1f}% {t1:>6.1f}%")

print(f"\n  ── Delta vs Bicubic ──")
bic = results['Bicubic']
for name, r in results.items():
    if name == 'Bicubic': continue
    s = lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}"
    print(f"  {name:<24}  AUC:{s(r['auc']-bic['auc'])}  "
          f"@10%:{s(r['tar10']-bic['tar10'])}  "
          f"@1%:{s(r['tar1']-bic['tar1'])}  "
          f"@0.1%:{s(r['tar01']-bic['tar01'])}")

print(f"\n  ── Delta vs CentreFace (best published AUC) ──")
cf_auc, cf_t10, cf_t1 = 94.8, 13.8, 3.1
for name, r in results.items():
    s = lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}"
    print(f"  {name:<24}  AUC:{s(r['auc']-cf_auc)}  "
          f"@10%:{s(r['tar10']-cf_t10)}  "
          f"@1%:{s(r['tar1']-cf_t1)}")

print(f"\n{'='*70}")
print(f"  Backbone: AdaFace IR-101, WebFace4M (381.6M params)")
print(f"  Reference: Kim et al., CVPR 2022")
print(f"{'='*70}\n")