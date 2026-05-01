"""
QMUL-SurvFace Evaluation Script
Evaluates Bicubic, AdaFace-SR v2 (synthetic), AdaFace-SR QMUL (real LR)
on the QMUL Face Identification Test Set (Rank-1, Rank-5, Rank-10)

Usage:
    python eval_qmul_survface.py \
        --qmul_dir data/QMUL-SurvFace \
        --adaface_v2  experiments/adaface_sr_v2/checkpoints/checkpoint_best.pth \
        --adaface_qmul experiments/adaface_qmul_v1/checkpoints/checkpoint_best.pth \
        --max_probes 2000 \
        > eval_qmul_results.txt 2>&1
"""

import os, sys, time, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from collections import defaultdict

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='QMUL-SurvFace Evaluation')
parser.add_argument('--qmul_dir',      required=True,  help='Path to QMUL-SurvFace root')
parser.add_argument('--adaface_v2',    required=True,  help='AdaFace-SR v2 checkpoint (synthetic trained)')
parser.add_argument('--adaface_qmul',  required=True,  help='AdaFace-SR QMUL checkpoint (real LR trained)')
parser.add_argument('--max_probes',    type=int, default=0, help='Limit probes for quick test (0=all)')
parser.add_argument('--batch_size',    type=int, default=32)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"  QMUL-SurvFace Evaluation")
print(f"  Device: {device}")
print(f"{'='*70}\n")

# ── Paths ─────────────────────────────────────────────────────────────────────
GALLERY_DIR     = os.path.join(args.qmul_dir, 'Face_Identification_Test_Set', 'gallery')
PROBE_DIR       = os.path.join(args.qmul_dir, 'Face_Identification_Test_Set', 'mated_probe')
GALLERY_MAT     = os.path.join(args.qmul_dir, 'Face_Identification_Test_Set', 'gallery_img_ID_pairs.mat')
PROBE_MAT       = os.path.join(args.qmul_dir, 'Face_Identification_Test_Set', 'mated_probe_img_ID_pairs.mat')

# ── Load .mat ID mappings ─────────────────────────────────────────────────────
def load_mat_pairs(mat_path):
    """Load image filename -> identity ID mapping from .mat file."""
    mat = loadmat(mat_path)
    # Try common key names used in QMUL
    for key in mat:
        if key.startswith('_'):
            continue
        val = mat[key]
        if hasattr(val, '__len__') and len(val) > 100:
            print(f"  Found key '{key}' with shape {np.array(val).shape}")
            return val
    raise ValueError(f"Cannot parse mat file: {mat_path}, keys={list(mat.keys())}")

print("[1/5] Loading QMUL identity mappings...")
try:
    gallery_mat = loadmat(GALLERY_MAT)
    probe_mat   = loadmat(PROBE_MAT)

    # QMUL mat files have 'gallery_img_ID_pairs' or similar
    # Structure: Nx2 where col0=filename, col1=ID
    def extract_pairs(mat):
        for k, v in mat.items():
            if k.startswith('_'): continue
            arr = np.array(v)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr
            if arr.ndim == 2 and arr.shape[0] == 2:
                return arr.T
        # Fallback: find largest 2D array
        best = None
        for k, v in mat.items():
            if k.startswith('_'): continue
            arr = np.array(v)
            if arr.ndim == 2 and (best is None or arr.size > best.size):
                best = arr
        return best

    gallery_pairs = extract_pairs(gallery_mat)
    probe_pairs   = extract_pairs(probe_mat)
    print(f"  Gallery pairs shape: {gallery_pairs.shape}")
    print(f"  Probe pairs shape:   {probe_pairs.shape}")

    # Extract filenames and IDs
    # Handles both string arrays and numeric arrays
    def parse_pairs(pairs):
        filenames, ids = [], []
        for row in pairs:
            try:
                fname = str(row[0]).strip()
                if hasattr(row[0], '__iter__') and not isinstance(row[0], str):
                    fname = str(row[0][0]).strip()
                fid = int(row[1]) if not hasattr(row[1], '__iter__') else int(row[1][0])
                filenames.append(fname)
                ids.append(fid)
            except Exception as e:
                continue
        return filenames, ids

    gallery_files, gallery_ids = parse_pairs(gallery_pairs)
    probe_files,   probe_ids   = parse_pairs(probe_pairs)

    print(f"  Gallery: {len(gallery_files)} images, {len(set(gallery_ids))} identities")
    print(f"  Probes:  {len(probe_files)} images, {len(set(probe_ids))} identities")

except Exception as e:
    print(f"  ERROR loading .mat files: {e}")
    print("  Falling back to filesystem-based loading...")
    # Fallback: parse identity from filename [PersonID]_[CameraID]_[ImageName].jpg
    gallery_files, gallery_ids = [], []
    for f in sorted(os.listdir(GALLERY_DIR)):
        if f.lower().endswith(('.jpg','.png','.jpeg')):
            pid = int(f.split('_')[0])
            gallery_files.append(f)
            gallery_ids.append(pid)

    probe_files, probe_ids = [], []
    for f in sorted(os.listdir(PROBE_DIR)):
        if f.lower().endswith(('.jpg','.png','.jpeg')):
            pid = int(f.split('_')[0])
            probe_files.append(f)
            probe_ids.append(pid)

    print(f"  Gallery: {len(gallery_files)} images, {len(set(gallery_ids))} identities")
    print(f"  Probes:  {len(probe_files)} images, {len(set(probe_ids))} identities")

# Optionally limit probes for speed
if args.max_probes > 0 and len(probe_files) > args.max_probes:
    idx = np.random.choice(len(probe_files), args.max_probes, replace=False)
    probe_files = [probe_files[i] for i in idx]
    probe_ids   = [probe_ids[i]   for i in idx]
    print(f"  (Limited to {args.max_probes} probes for speed)")

# ── Load FR backbone (ArcFace iResNet50, CPU to save GPU for SR) ──────────────
print("\n[2/5] Loading FR backbone...")
sys.path.insert(0, '/raid/home/dgxuser8/capstone1/version1')

from src.models.iresnet import iresnet50
fr_model = iresnet50()
fr_ckpt_path = '/raid/home/dgxuser8/capstone1/version1/pretrained/arcface_pytorch/iresnet50.pth'
if os.path.exists(fr_ckpt_path):
    state = torch.load(fr_ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    fr_model.load_state_dict(state, strict=False)
    print(f"  ✅ FR backbone loaded from {fr_ckpt_path}")
else:
    print(f"  ⚠️  FR backbone not found at {fr_ckpt_path}, using random weights")
fr_model = fr_model.to(device).eval()

def extract_embedding(img_112):
    """Extract 512-dim ArcFace embedding from 112x112 BGR image."""
    img = cv2.cvtColor(img_112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    t = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = fr_model(t)
        emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()[0]

# ── Load SR models ────────────────────────────────────────────────────────────
print("\n[3/5] Loading SR models...")
from src.models.adaface_sr import AdaFaceSR

def load_adaface(ckpt_path, name):
    model = AdaFaceSR().to(device).eval()
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        epoch = ckpt.get('epoch', '?')
        alpha_info = ""
        print(f"  ✅ {name} loaded (epoch={epoch})")
        return model
    except Exception as e:
        print(f"  ❌ {name} failed: {e}")
        return None

adaface_v2   = load_adaface(args.adaface_v2,   'AdaFace-SR v2 (synthetic)')
adaface_qmul = load_adaface(args.adaface_qmul, 'AdaFace-SR QMUL (real LR)')

def enhance_image(img_bgr, sr_model, target_size=112):
    """Apply SR enhancement and return 112x112 BGR image."""
    # Bicubic upsample to 112x112
    bic = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    if sr_model is None:
        return bic

    # Convert to tensor
    img_rgb = cv2.cvtColor(bic, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb.transpose(2,0,1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Detect input resolution for resolution encoder
        h, w = img_bgr.shape[:2]
        res = torch.tensor([min(h, w)], dtype=torch.long).to(device)
        try:
            out = sr_model(t, res)
        except TypeError:
            out = sr_model(t)
        # Model may return (output, alpha) tuple
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1,2,0)

    out_bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out_bgr

# ── Helper: load and resize image ────────────────────────────────────────────
def load_image(path, size=None):
    img = cv2.imread(path)
    if img is None:
        return None
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img

# ── Extract gallery embeddings (same for all methods — gallery is HR) ─────────
print("\n[4/5] Extracting gallery embeddings...")

# Always rebuild from filesystem — most reliable approach
# QMUL filename format: [PersonID]_[CameraID]_[ImageName].jpg
print("  Building gallery from filesystem (filename-based ID parsing)...")
gallery_files_fs, gallery_ids_fs = [], []
actual_files = sorted(os.listdir(GALLERY_DIR))
print(f"  Found {len(actual_files)} files in gallery dir")
print(f"  Sample filenames: {actual_files[:3]}")

for f in actual_files:
    if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    try:
        pid = int(f.split('_')[0])
        gallery_files_fs.append(f)
        gallery_ids_fs.append(pid)
    except (ValueError, IndexError):
        continue

print(f"  Parsed {len(gallery_files_fs)} gallery images, {len(set(gallery_ids_fs))} identities")

# Use filesystem version (guaranteed to match actual files)
gallery_files = gallery_files_fs
gallery_ids   = gallery_ids_fs

# Also rebuild probe list from filesystem for consistency
print("  Building probe list from filesystem...")
probe_files_fs, probe_ids_fs = [], []
for f in sorted(os.listdir(PROBE_DIR)):
    if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    try:
        pid = int(f.split('_')[0])
        probe_files_fs.append(f)
        probe_ids_fs.append(pid)
    except (ValueError, IndexError):
        continue

print(f"  Parsed {len(probe_files_fs)} probe images, {len(set(probe_ids_fs))} identities")
probe_files = probe_files_fs
probe_ids   = probe_ids_fs

# Optionally limit probes
if args.max_probes > 0 and len(probe_files) > args.max_probes:
    idx = np.random.choice(len(probe_files), args.max_probes, replace=False)
    probe_files = [probe_files[i] for i in idx]
    probe_ids   = [probe_ids[i]   for i in idx]
    print(f"  (Limited to {args.max_probes} probes)")

# Extract embeddings
gallery_embs = {}
failed_gallery = 0
for i, (fname, fid) in enumerate(zip(gallery_files, gallery_ids)):
    fpath = os.path.join(GALLERY_DIR, fname)
    img = load_image(fpath, size=112)
    if img is None:
        failed_gallery += 1
        continue
    emb = extract_embedding(img)
    if fid not in gallery_embs:
        gallery_embs[fid] = []
    gallery_embs[fid].append(emb)
    if (i+1) % 5000 == 0:
        print(f"  Gallery: {i+1}/{len(gallery_files)} done...")

print(f"  Loaded embeddings for {len(gallery_embs)} identities, failed={failed_gallery}")

if len(gallery_embs) == 0:
    raise RuntimeError("No gallery images loaded — check GALLERY_DIR path and file permissions")

# Average pool per identity and normalise
gallery_ids_list   = sorted(gallery_embs.keys())
gallery_emb_matrix = np.stack([
    np.mean(gallery_embs[fid], axis=0) for fid in gallery_ids_list
])  # shape: (n_ids, 512)
norms = np.linalg.norm(gallery_emb_matrix, axis=1, keepdims=True) + 1e-8
gallery_emb_matrix = gallery_emb_matrix / norms
print(f"  Gallery matrix: {gallery_emb_matrix.shape}")
print(f"  Failed to load: {failed_gallery} gallery images")

# ── Evaluate each SR method ───────────────────────────────────────────────────
print("\n[5/5] Evaluating SR methods on probes...")

methods = [
    ('Bicubic',            None),
    ('AdaFace-SR v2',      adaface_v2),
    ('AdaFace-SR QMUL',    adaface_qmul),
]

results = {}

for method_name, sr_model in methods:
    print(f"\n  ── {method_name} ──")
    rank1, rank5, rank10 = 0, 0, 0
    total, failed = 0, 0
    t0 = time.time()

    for i, (fname, true_id) in enumerate(zip(probe_files, probe_ids)):
        fpath = os.path.join(PROBE_DIR, fname)
        img = load_image(fpath)  # load at native resolution
        if img is None:
            failed += 1
            continue

        # Apply SR enhancement
        enhanced = enhance_image(img, sr_model, target_size=112)

        # Extract embedding
        emb = extract_embedding(enhanced)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        # Cosine similarity against gallery
        sims = gallery_emb_matrix @ emb  # shape: (n_gallery_ids,)
        ranked = np.argsort(-sims)       # descending

        # Check if true_id is in gallery
        if true_id not in gallery_ids_list:
            failed += 1
            continue

        true_pos = gallery_ids_list.index(true_id)
        rank = np.where(ranked == true_pos)[0][0] + 1  # 1-indexed

        if rank == 1:  rank1  += 1
        if rank <= 5:  rank5  += 1
        if rank <= 10: rank10 += 1
        total += 1

        if (i+1) % 500 == 0:
            elapsed = time.time() - t0
            r1_so_far = rank1/total*100 if total > 0 else 0
            print(f"    [{i+1}/{len(probe_files)}] Rank-1={r1_so_far:.1f}%  ({elapsed:.0f}s)")

    if total > 0:
        r1  = rank1  / total * 100
        r5  = rank5  / total * 100
        r10 = rank10 / total * 100
    else:
        r1 = r5 = r10 = 0.0

    elapsed = time.time() - t0
    print(f"    FINAL → Rank-1={r1:.1f}%  Rank-5={r5:.1f}%  Rank-10={r10:.1f}%")
    print(f"    Total={total}  Failed={failed}  Time={elapsed:.0f}s")
    results[method_name] = {'rank1': r1, 'rank5': r5, 'rank10': r10,
                             'total': total, 'failed': failed}

# ── Final Report ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  QMUL-SurvFace FINAL RESULTS")
print(f"  (Mated probe, {len(probe_files)} probes, {len(gallery_ids_list)} gallery identities)")
print(f"{'='*70}")
print(f"  {'Method':<22} {'Rank-1':>8} {'Rank-5':>8} {'Rank-10':>8} {'Total':>8}")
print(f"  {'-'*60}")
for name, r in results.items():
    print(f"  {name:<22} {r['rank1']:>7.1f}% {r['rank5']:>7.1f}% {r['rank10']:>7.1f}% {r['total']:>8}")

print(f"\n  ── Delta vs Bicubic ──")
bic = results.get('Bicubic', {})
for name, r in results.items():
    if name == 'Bicubic': continue
    d1  = r['rank1']  - bic.get('rank1',  0)
    d5  = r['rank5']  - bic.get('rank5',  0)
    d10 = r['rank10'] - bic.get('rank10', 0)
    sign = lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}"
    print(f"  {name:<22} {sign(d1):>8} {sign(d5):>8} {sign(d10):>8}")

print(f"\n{'='*70}")
print("  KEY: Positive Δ = beats bicubic (good), Negative Δ = worse than bicubic")
print(f"{'='*70}\n")