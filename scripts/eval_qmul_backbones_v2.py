#!/usr/bin/env python3
"""
eval_qmul_backbones.py  —  Backbone independence on QMUL-SurvFace.

Tests AdaFace-SR v2 and Bicubic baseline with THREE FR backbones:
  1. AdaFace IR-101    (adaface_ir101_webface4m.ckpt)   -- already in paper
  2. ArcFace iResNet-50 (iresnet50.pth)                 -- NEW
  3. InsightFace buffalo_l w600k_r50 (ONNX)             -- NEW

Key fixes vs old version:
  - AdaFace-SR forward returns TUPLE (sr_t, alpha_t) — handle correctly
  - Model input: BGR→RGB /255.0  (NO mean/std normalisation)
  - Model output: clamp(-1,1) then *0.5+0.5 denorm (range [-4,4] confirmed)
  - buffalo_l: input normalised to [-1,1]  (matches w600k training)
  - iResNet-50: input normalised to [-1,1]  (matches ArcFace training)
  - AdaFace IR-101: same as iResNet-50

Usage:
  cd /raid/home/dgxuser8/capstone1/version1
  source ../capstone1/bin/activate
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_qmul_backbones.py \
      2>&1 | tee eval_backbone_ablation.txt
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import scipy.io
import onnxruntime as ort
from pathlib import Path
from sklearn.metrics import roc_curve, auc as sk_auc

ROOT     = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRETRAIN = ROOT / 'pretrained'
QMUL_DIR = ROOT / 'data' / 'QMUL-SurvFace'
SR_CKPT  = ROOT / 'experiments/adaface_sr_v2/checkpoints/checkpoint_best.pth'
sys.path.insert(0, str(ROOT))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# QMUL PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════
def load_qmul_pairs():
    """Load official positive/negative pair filenames from .mat files."""
    proto_dir = QMUL_DIR / 'Face_Verification_Test_Set'
    pos_mat   = proto_dir / 'positive_pairs_names.mat'
    neg_mat   = proto_dir / 'negative_pairs_names.mat'

    if not pos_mat.exists():
        raise FileNotFoundError(f"QMUL protocol not found: {pos_mat}")

    def parse_mat(path):
        data = scipy.io.loadmat(str(path))
        key  = [k for k in data if not k.startswith('_')][0]
        raw  = data[key]
        pairs = []
        for i in range(raw.shape[0]):
            try:
                a = str(raw[i, 0][0]) if raw[i, 0].size > 0 else ''
                b = str(raw[i, 1][0]) if raw[i, 1].size > 0 else ''
                if a and b:
                    pairs.append((a, b))
            except Exception:
                continue
        return pairs

    pos = parse_mat(pos_mat)
    neg = parse_mat(neg_mat)
    log(f"  Protocol: {len(pos)} positive + {len(neg)} negative pairs")
    return pos, neg

def load_qmul_image_bgr(filename):
    """Load QMUL image as BGR numpy array."""
    # Confirmed path from diagnostic output:
    # images live in Face_Verification_Test_Set/verification_images/
    img_dir = QMUL_DIR / 'Face_Verification_Test_Set' / 'verification_images'
    fname   = Path(filename).name

    direct = img_dir / fname
    if direct.exists():
        return cv2.imread(str(direct))

    # Fallback: recursive search
    matches = list(img_dir.rglob(fname))
    if matches:
        return cv2.imread(str(matches[0]))
    return None

# ══════════════════════════════════════════════════════════════════════════════
# AdaFace-SR ENHANCEMENT
# Matches evaluate_full.py SRMethodAdaFaceSR.enhance() exactly
# ══════════════════════════════════════════════════════════════════════════════
def load_adaface_sr():
    """
    Load AdaFace-SR v2 from checkpoint.
    Confirmed: src.models.adaface_sr, AdaFaceSR(config), forward returns tuple.
    """
    from src.models.adaface_sr import AdaFaceSR
    ckpt = torch.load(str(SR_CKPT), map_location='cpu')
    cfg  = ckpt.get('config', {})
    model_config = {
        'num_feat':      cfg.get('num_feat',      64),
        'num_block':     cfg.get('num_block',      4),
        'num_grow':      cfg.get('num_grow',       32),
        'res_embed_dim': cfg.get('res_embed_dim',  64),
        'gate_channels': cfg.get('gate_channels',  32),
    }
    model = AdaFaceSR(model_config)
    state = ckpt.get('model_state_dict', ckpt)
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        log(f"  WARN: {len(missing)} missing keys")
    model = model.to(DEVICE).eval()
    ep = ckpt.get('epoch', '?')
    bl = ckpt.get('best_loss', '?')
    bl_str = f"{bl:.4f}" if isinstance(bl, float) else str(bl)
    log(f"  AdaFace-SR loaded: epoch={ep}  best_loss={bl_str}")
    return model

@torch.no_grad()
def adaface_sr_enhance_bgr(model, bgr):
    """
    Apply AdaFace-SR v2.
    Confirmed from debug_adaface_forward.py:
      - Input: BGR→RGB /255.0 (no mean/std — matches evaluate_full.py)
      - model(x, lr_size) returns TUPLE (sr_t, alpha_t)
      - out[0] range is [-3.8, 4.0] (normalised with mean=0.5 std=0.5)
      - Denorm: clamp(-1,1) → *0.5+0.5 → [0,1] → *255 → BGR
    """
    h, w    = bgr.shape[:2]
    lr_size = float(min(h, w))

    # Bicubic to 112×112
    lr_112 = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_CUBIC)

    # BGR→RGB /255.0 (matches evaluate_full.py SRMethodAdaFaceSR.enhance line 332)
    rgb    = cv2.cvtColor(lr_112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lr_t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    size_t = torch.tensor([lr_size], device=DEVICE)

    # Forward — confirmed TUPLE (sr_t, alpha_t), out[0] in [-4, 4]
    output = model(lr_t, size_t)
    sr_t   = output[0] if isinstance(output, (tuple, list)) else output

    # Denormalise: network was trained with mean=0.5 std=0.5 normalisation
    # So output is in normalised space; undo: clamp→*0.5+0.5
    sr_t   = sr_t.clamp(-1, 1) * 0.5 + 0.5   # → [0, 1]
    sr_t   = sr_t.clamp(0, 1)

    sr_rgb = sr_t[0].permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor((sr_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def bicubic_enhance_bgr(bgr):
    """Bicubic resize to 112×112."""
    return cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_CUBIC)

# ══════════════════════════════════════════════════════════════════════════════
# FR BACKBONES
# All three produce 512-dim L2-normalised embeddings from 112×112 BGR input.
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared ONNX normalisation helper ─────────────────────────────────────────
def _bgr_to_input(bgr_112):
    """Convert 112×112 BGR to (1,3,112,112) tensor normalised to [-1,1]."""
    rgb = cv2.cvtColor(bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp = ((rgb / 255.0) - 0.5) / 0.5       # → [-1, 1]
    return inp.transpose(2, 0, 1)[np.newaxis] # NHWC → NCHW

def _l2_norm(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

# ── Backbone 1: AdaFace IR-101 ───────────────────────────────────────────────
class AdaFaceIR101:
    name = 'AdaFace IR-101'

    def __init__(self):
        ckpt_path = PRETRAIN / 'recognition/adaface_ir101_webface4m.ckpt'
        sys.path.insert(0, str(ROOT))

        # Confirmed: src.models.iresnet has iresnet100()
        from src.models.iresnet import iresnet100
        model = iresnet100()
        ckpt  = torch.load(str(ckpt_path), map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        state = {k.replace('model.', ''): v for k, v in state.items()
                 if not k.startswith('head.')}
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            log(f"  [{self.name}] WARN: {len(missing)} missing keys")
        self.model = model.to(DEVICE).eval()
        n = sum(p.numel() for p in model.parameters()) / 1e6
        log(f"  [{self.name}] Loaded ({n:.1f}M params)")

    @torch.no_grad()
    def embed(self, bgr_112):
        inp = torch.from_numpy(_bgr_to_input(bgr_112)).to(DEVICE)
        out = self.model(inp)
        emb = (out[0] if isinstance(out, (tuple, list)) else out).squeeze(0).cpu().numpy()
        emb = emb.astype(np.float32)
        if not np.all(np.isfinite(emb)):
            return None  # architecture mismatch → NaN outputs
        return _l2_norm(emb)

# ── Backbone 2: ArcFace iResNet-50 ──────────────────────────────────────────
class ArcFaceIResNet50:
    name = 'ArcFace iResNet-50'

    def __init__(self):
        ckpt_path = PRETRAIN / 'arcface_pytorch/iresnet50.pth'

        # Confirmed: src.models.iresnet has iresnet50()
        from src.models.iresnet import iresnet50
        model = iresnet50()
        ckpt  = torch.load(str(ckpt_path), map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            log(f"  [{self.name}] WARN: {len(missing)} missing keys")
        self.model = model.to(DEVICE).eval()
        n = sum(p.numel() for p in model.parameters()) / 1e6
        log(f"  [{self.name}] Loaded ({n:.1f}M params)")

    @torch.no_grad()
    def embed(self, bgr_112):
        inp = torch.from_numpy(_bgr_to_input(bgr_112)).to(DEVICE)
        out = self.model(inp)
        emb = (out[0] if isinstance(out, (tuple, list)) else out).squeeze(0).cpu().numpy()
        return _l2_norm(emb.astype(np.float32))

# ── Backbone 3: InsightFace buffalo_l (direct ONNX, no detection crash) ─────
class BuffaloLONNX:
    name = 'buffalo_l (w600k_r50)'

    def __init__(self):
        path = str(PRETRAIN / 'recognition/models/buffalo_l/w600k_r50.onnx')
        self.sess = ort.InferenceSession(
            path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.inp_name = self.sess.get_inputs()[0].name
        log(f"  [{self.name}] ONNX loaded  (direct inference, no detection)")

    def embed(self, bgr_112):
        # w600k_r50: input (1,3,112,112) in [-1,1]  (same as ArcFace training)
        inp = _bgr_to_input(bgr_112)
        out = self.sess.run(None, {self.inp_name: inp})[0][0].astype(np.float32)
        return _l2_norm(out)

# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def run_verification(backbone, enhance_fn, pos_pairs, neg_pairs, bb_name, method_name):
    """
    Run full QMUL verification protocol.
    Returns dict with AUC and TAR@FAR metrics.
    """
    scores = []
    labels = []
    skipped = 0
    total   = len(pos_pairs) + len(neg_pairs)
    all_pairs = [(p, 1) for p in pos_pairs] + [(p, 0) for p in neg_pairs]

    tag = f"{bb_name} + {method_name}"
    log(f"  [{tag}] {total} pairs...")

    for idx, ((fn_a, fn_b), label) in enumerate(all_pairs):
        if idx % 1000 == 0 and idx > 0:
            log(f"    {idx}/{total}  ({100*idx//total}%)  skipped={skipped}")

        bgr_a = load_qmul_image_bgr(fn_a)
        bgr_b = load_qmul_image_bgr(fn_b)

        if bgr_a is None or bgr_b is None:
            skipped += 1
            continue

        try:
            enh_a = enhance_fn(bgr_a)
            enh_b = enhance_fn(bgr_b)
            emb_a = backbone.embed(enh_a)
            emb_b = backbone.embed(enh_b)
        except Exception as e:
            skipped += 1
            continue

        if emb_a is None or emb_b is None:
            skipped += 1
            continue

        score = float(np.dot(emb_a, emb_b))
        if not np.isfinite(score):
            skipped += 1
            continue
        scores.append(score)
        labels.append(label)

    if len(scores) < 100:
        log(f"    WARNING: only {len(scores)} pairs evaluated")
        return None

    scores = np.array(scores)
    labels = np.array(labels)

    # Filter out NaN scores (caused by backbone architecture mismatch)
    valid = np.isfinite(scores)
    n_nan = (~valid).sum()
    if n_nan > 0:
        log(f"    WARNING: {n_nan} NaN scores filtered out ({n_nan/len(scores)*100:.1f}%)")
    scores = scores[valid]
    labels = labels[valid]

    if len(scores) < 100:
        log(f"    ERROR: only {len(scores)} valid scores after NaN filter")
        return None

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = sk_auc(fpr, tpr) * 100

    tar = {}
    for target_far in [0.30, 0.10, 0.01, 0.001]:
        idx     = np.searchsorted(fpr, target_far, side='right') - 1
        idx     = max(0, min(idx, len(tpr) - 1))
        tar[target_far] = tpr[idx] * 100

    gen_scores = scores[labels == 1]
    imp_scores = scores[labels == 0]
    gap        = gen_scores.mean() - imp_scores.mean()

    result = {
        'backbone': bb_name,
        'method':   method_name,
        'n_pairs':  len(scores),
        'skipped':  skipped,
        'AUC':      roc_auc,
        'TAR@30':   tar[0.30],
        'TAR@10':   tar[0.10],
        'TAR@1':    tar[0.01],
        'TAR@0.1':  tar[0.001],
        'gen_mean': float(gen_scores.mean()),
        'imp_mean': float(imp_scores.mean()),
        'gap':      float(gap),
    }

    log(f"    Done: AUC={roc_auc:.1f}%  @10%={tar[0.10]:.1f}%  "
        f"@1%={tar[0.01]:.1f}%  @0.1%={tar[0.001]:.1f}%  "
        f"gap={gap:.3f}  skipped={skipped}")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log("="*65)
    log("QMUL-SurvFace Backbone Independence Evaluation")
    log("="*65)
    log(f"Device: {DEVICE}")

    # Load pairs
    log("\nLoading QMUL-SurvFace pairs...")
    pos_pairs, neg_pairs = load_qmul_pairs()

    # Load SR model
    log("\nLoading AdaFace-SR v2...")
    sr_model = load_adaface_sr()

    # SR methods
    methods = {
        'Bicubic':    lambda bgr: bicubic_enhance_bgr(bgr),
        'AdaFace-SR': lambda bgr: adaface_sr_enhance_bgr(sr_model, bgr),
    }

    # Load backbones
    backbones = []

    log("\nLoading FR backbones...")

    try:
        bb = AdaFaceIR101()
        backbones.append(bb)
    except Exception as e:
        log(f"  SKIP AdaFace IR-101: {e}")

    try:
        bb = ArcFaceIResNet50()
        backbones.append(bb)
    except Exception as e:
        log(f"  SKIP iResNet-50: {e}")

    try:
        bb = BuffaloLONNX()
        backbones.append(bb)
    except Exception as e:
        log(f"  SKIP buffalo_l: {e}")

    if not backbones:
        log("ERROR: No backbones loaded.")
        return

    # Run all combinations
    all_results = []
    for backbone in backbones:
        for method_name, enhance_fn in methods.items():
            log(f"\n── {backbone.name} + {method_name} ──")
            result = run_verification(
                backbone, enhance_fn,
                pos_pairs, neg_pairs,
                backbone.name, method_name)
            if result:
                all_results.append(result)

    # Print table
    SEP = "="*75
    print(f"\n{SEP}")
    print("BACKBONE INDEPENDENCE — QMUL-SurvFace Face Verification")
    print("Paste into Table XVI (tab:backbone_ablation) in journal_paper_v8.tex")
    print(SEP)

    header = f"{'Backbone':<26} {'Method':<14} {'AUC':>6} {'@30%':>7} {'@10%':>7} {'@1%':>7} {'@0.1%':>7} {'Gap':>7}"
    print(header)
    print("-"*75)

    # Published baselines
    print("[Official leaderboard (2016-2017):]")
    published = [
        ("CentreFace",  94.8, 27.3, 13.8, 3.1, "--"),
        ("FaceNet",     93.5, 12.7,  4.3, 1.0, "--"),
        ("SphereFace",  83.9, 21.3,  8.3, 1.0, "--"),
    ]
    for name, auc, t30, t10, t1, t01 in published:
        print(f"  {name:<38} {auc:>6.1f} {t30:>7.1f} {t10:>7.1f} {t1:>7.1f} {str(t01):>7}")

    print("\n[This work:]")
    for r in all_results:
        gap_str = f"{r['gap']:.3f}"
        print(f"  {r['backbone']:<26} {r['method']:<14} "
              f"{r['AUC']:>6.1f} {r['TAR@30']:>7.1f} {r['TAR@10']:>7.1f} "
              f"{r['TAR@1']:>7.1f} {r['TAR@0.1']:>7.1f} {gap_str:>7}")

    print(SEP)

    # AdaFace-SR delta over Bicubic
    print("\nAdaFace-SR gain over Bicubic (TAR@1%):")
    backbone_results = {}
    for r in all_results:
        backbone_results.setdefault(r['backbone'], {})[r['method']] = r
    for bb_name, methods_r in backbone_results.items():
        bic = methods_r.get('Bicubic', {}).get('TAR@1', None)
        asr = methods_r.get('AdaFace-SR', {}).get('TAR@1', None)
        if bic is not None and asr is not None:
            print(f"  {bb_name:<26}: {bic:.1f}% → {asr:.1f}%  ({asr-bic:+.1f}pp)")

    # Score gap analysis
    print("\nScore gap (genuine - impostor mean cosine similarity):")
    for r in all_results:
        print(f"  {r['backbone']:<26} {r['method']:<14}: "
              f"gen={r['gen_mean']:.3f}  imp={r['imp_mean']:.3f}  "
              f"gap={r['gap']:.3f}")

    # Save
    out_path = ROOT / 'eval_backbone_ablation.txt'
    with open(str(out_path), 'w') as f:
        f.write(f"{SEP}\nBACKBONE INDEPENDENCE — QMUL-SurvFace\n{SEP}\n")
        f.write(header + "\n" + "-"*75 + "\n")
        for r in all_results:
            f.write(f"{r['backbone']:<26} {r['method']:<14} "
                    f"{r['AUC']:>6.1f} {r['TAR@30']:>7.1f} {r['TAR@10']:>7.1f} "
                    f"{r['TAR@1']:>7.1f} {r['TAR@0.1']:>7.1f} {r['gap']:>7.3f}\n")
    log(f"\nSaved → {out_path}")

if __name__ == '__main__':
    main()
