"""
QMUL-SurvFace Face Verification Evaluation
Directly comparable to the official leaderboard:
  https://qmul-survface.github.io/benchmark.html

Metrics: AUC, TAR@FAR=30%, TAR@FAR=10%, TAR@FAR=1%, TAR@FAR=0.1%

Usage:
    python scripts/eval_qmul_verification.py \
        --qmul_dir data/QMUL-SurvFace \
        --adaface_v2   experiments/adaface_sr_v2/checkpoints/checkpoint_best.pth \
        --adaface_qmul experiments/adaface_qmul_v1/checkpoints/checkpoint_best.pth \
        > eval_qmul_verification.txt 2>&1
"""

import os, sys, time, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, roc_curve

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='QMUL-SurvFace Verification Eval')
parser.add_argument('--qmul_dir',     required=True)
parser.add_argument('--adaface_v2',   required=True, help='Synthetic-trained AdaFace-SR checkpoint')
parser.add_argument('--adaface_qmul', required=True, help='QMUL-trained AdaFace-SR checkpoint')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n{'='*70}")
print(f"  QMUL-SurvFace Face Verification Evaluation")
print(f"  Comparable to official leaderboard (CentreFace/SphereFace/FaceNet)")
print(f"  Device: {device}")
print(f"{'='*70}\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
VER_DIR  = os.path.join(args.qmul_dir, 'Face_Verification_Test_Set', 'verification_images')
POS_MAT  = os.path.join(args.qmul_dir, 'Face_Verification_Test_Set', 'positive_pairs_names.mat')
NEG_MAT  = os.path.join(args.qmul_dir, 'Face_Verification_Test_Set', 'negative_pairs_names.mat')

# ── Load pairs ────────────────────────────────────────────────────────────────
print("[1/5] Loading verification pairs...")
pos_mat = loadmat(POS_MAT)
neg_mat = loadmat(NEG_MAT)

def parse_pairs_mat(mat):
    """Parse Nx2 object array of filenames into list of (img1, img2) tuples."""
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
    raise ValueError(f"Cannot parse mat, keys={list(mat.keys())}")

pos_pairs = parse_pairs_mat(pos_mat)  # 5320 genuine pairs
neg_pairs = parse_pairs_mat(neg_mat)  # 5320 impostor pairs

print(f"  Positive (genuine) pairs:  {len(pos_pairs)}")
print(f"  Negative (impostor) pairs: {len(neg_pairs)}")
print(f"  Total verification images: {len(os.listdir(VER_DIR))}")
print(f"  Sample positive: {pos_pairs[0]}")
print(f"  Sample negative: {neg_pairs[0]}")

# ── Load FR backbone ──────────────────────────────────────────────────────────
print("\n[2/5] Loading ArcFace FR backbone...")
sys.path.insert(0, '/raid/home/dgxuser8/capstone1/version1')

from src.models.iresnet import iresnet50
fr_model = iresnet50()
fr_ckpt  = '/raid/home/dgxuser8/capstone1/version1/pretrained/arcface_pytorch/iresnet50.pth'

if os.path.exists(fr_ckpt):
    state = torch.load(fr_ckpt, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    fr_model.load_state_dict(state, strict=False)
    print(f"  ✅ ArcFace iResNet-50 loaded")
else:
    print(f"  ⚠️  Checkpoint not found at {fr_ckpt} — using random weights")

fr_model = fr_model.to(device).eval()

def extract_embedding(img_bgr):
    """Extract L2-normalised 512-dim ArcFace embedding from BGR image."""
    img = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    t   = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = fr_model(t)
        emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()[0]

# ── Load SR models ────────────────────────────────────────────────────────────
print("\n[3/5] Loading SR models...")
from src.models.adaface_sr import AdaFaceSR

def load_adaface(path, name):
    model = AdaFaceSR().to(device).eval()
    try:
        ckpt  = torch.load(path, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        ep = ckpt.get('epoch', '?')
        print(f"  ✅ {name} loaded (epoch={ep})")
        return model
    except Exception as e:
        print(f"  ❌ {name} failed: {e}")
        return None

adaface_v2   = load_adaface(args.adaface_v2,   'AdaFace-SR v2 (synthetic)')
adaface_qmul = load_adaface(args.adaface_qmul, 'AdaFace-SR QMUL (real LR)')

def enhance(img_bgr, sr_model):
    """SR-enhance image. Returns 112x112 BGR. Falls back to bicubic if model=None."""
    bic = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
    if sr_model is None:
        return bic
    rgb = cv2.cvtColor(bic, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float().to(device)
    h, w = img_bgr.shape[:2]
    res  = torch.tensor([min(h, w)], dtype=torch.long).to(device)
    with torch.no_grad():
        try:
            out = sr_model(t, res)
        except TypeError:
            out = sr_model(t)
        out = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# ── Cache all embeddings per method ──────────────────────────────────────────
print("\n[4/5] Extracting embeddings for all verification images...")

all_images = set()
for f1, f2 in pos_pairs + neg_pairs:
    all_images.add(f1)
    all_images.add(f2)
all_images = sorted(all_images)
print(f"  Unique images to process: {len(all_images)}")

methods = [
    ('Bicubic',          None),
    ('AdaFace-SR v2',    adaface_v2),
    ('AdaFace-SR QMUL',  adaface_qmul),
]

method_embeddings = {}  # method_name -> {filename -> embedding}

for method_name, sr_model in methods:
    print(f"\n  Processing: {method_name}")
    emb_cache = {}
    failed    = 0
    t0        = time.time()

    for i, fname in enumerate(all_images):
        fpath = os.path.join(VER_DIR, fname)
        img   = cv2.imread(fpath)
        if img is None:
            failed += 1
            emb_cache[fname] = None
            continue

        enhanced        = enhance(img, sr_model)
        emb_cache[fname] = extract_embedding(enhanced)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(all_images)}] {elapsed:.0f}s elapsed...")

    method_embeddings[method_name] = emb_cache
    print(f"  Done. Failed to load: {failed} images. Time: {time.time()-t0:.0f}s")

# ── Compute verification scores ───────────────────────────────────────────────
print("\n[5/5] Computing verification scores and metrics...")

def tar_at_far(labels, scores, far_target):
    """Compute TAR at a given FAR threshold."""
    fpr, tpr, _ = roc_curve(labels, scores)
    # Find closest FAR
    idx = np.searchsorted(fpr, far_target, side='right') - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx]) * 100

FAR_TARGETS = [0.30, 0.10, 0.01, 0.001]  # 30%, 10%, 1%, 0.1%

results = {}

for method_name in [m[0] for m in methods]:
    embs   = method_embeddings[method_name]
    scores = []
    labels = []
    skipped = 0

    # Genuine pairs → label=1
    for f1, f2 in pos_pairs:
        e1 = embs.get(f1)
        e2 = embs.get(f2)
        if e1 is None or e2 is None:
            skipped += 1
            continue
        cos = float(np.dot(e1, e2))
        scores.append(cos)
        labels.append(1)

    # Impostor pairs → label=0
    for f1, f2 in neg_pairs:
        e1 = embs.get(f1)
        e2 = embs.get(f2)
        if e1 is None or e2 is None:
            skipped += 1
            continue
        cos = float(np.dot(e1, e2))
        scores.append(cos)
        labels.append(0)

    labels = np.array(labels)
    scores = np.array(scores)

    auc  = roc_auc_score(labels, scores) * 100
    tars = [tar_at_far(labels, scores, f) for f in FAR_TARGETS]

    results[method_name] = {
        'auc':     auc,
        'tar30':   tars[0],
        'tar10':   tars[1],
        'tar1':    tars[2],
        'tar01':   tars[3],
        'n_pairs': len(scores),
        'skipped': skipped,
    }
    print(f"  {method_name}: AUC={auc:.1f}%  TAR@1%={tars[2]:.1f}%  (skipped={skipped})")

# ── Final Report ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  QMUL-SurvFace VERIFICATION RESULTS")
print(f"  (5320 genuine + 5320 impostor pairs)")
print(f"{'='*70}")
print(f"  {'Method':<22} {'AUC':>7} {'@30%':>7} {'@10%':>7} {'@1%':>7} {'@0.1%':>7}")
print(f"  {'-'*65}")

# Published baselines from official leaderboard
baselines = {
    'CentreFace (ECCV16)': {'auc':94.8,'tar30':27.3,'tar10':13.8,'tar1':3.1,'tar01':None},
    'SphereFace (CVPR17)': {'auc':83.9,'tar30':21.3,'tar10':8.3, 'tar1':1.0,'tar01':None},
    'FaceNet (CVPR15)':    {'auc':93.5,'tar30':12.7,'tar10':4.3, 'tar1':1.0,'tar01':None},
}

for name, r in results.items():
    t01 = f"{r['tar01']:>6.1f}%" if r['tar01'] is not None else '     -'
    print(f"  {name:<22} {r['auc']:>6.1f}% {r['tar30']:>6.1f}% "
          f"{r['tar10']:>6.1f}% {r['tar1']:>6.1f}% {t01}")

print(f"\n  ── Published SOTA (official leaderboard) ──")
for name, r in baselines.items():
    t01 = f"{'N/A':>7}"
    print(f"  {name:<22} {r['auc']:>6.1f}% {r['tar30']:>6.1f}% "
          f"{r['tar10']:>6.1f}% {r['tar1']:>6.1f}% {t01}")

print(f"\n  ── Delta vs Bicubic ──")
bic = results.get('Bicubic', {})
for name, r in results.items():
    if name == 'Bicubic':
        continue
    da   = r['auc']   - bic.get('auc',   0)
    d30  = r['tar30'] - bic.get('tar30', 0)
    d10  = r['tar10'] - bic.get('tar10', 0)
    d1   = r['tar1']  - bic.get('tar1',  0)
    d01  = r['tar01'] - bic.get('tar01', 0)
    sign = lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}"
    print(f"  {name:<22} {sign(da):>7} {sign(d30):>7} {sign(d10):>7} "
          f"{sign(d1):>7} {sign(d01):>7}")

print(f"\n  ── Delta vs CentreFace (best published) ──")
cf = baselines['CentreFace (ECCV16)']
for name, r in results.items():
    da  = r['auc']   - cf['auc']
    d10 = r['tar10'] - cf['tar10']
    d1  = r['tar1']  - cf['tar1']
    sign = lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}"
    print(f"  {name:<22}  AUC:{sign(da)}  TAR@10%:{sign(d10)}  TAR@1%:{sign(d1)}")

print(f"\n{'='*70}")
print("  NOTE: TAR@FAR metrics match the official QMUL leaderboard protocol")
print("  Positive delta vs CentreFace = beats published SOTA on this benchmark")
print(f"{'='*70}\n")
