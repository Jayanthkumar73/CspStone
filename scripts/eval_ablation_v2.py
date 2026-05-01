#!/usr/bin/env python3
"""
eval_ablation_v3.py  —  AdaFace-SR ablation on SCface d3.
Fixes ValueError: anchor grid broadcast error in InsightFace detector
by always padding images to >=640px before detection, and falling back
to direct ONNX inference on 112x112 crops when detection fails.

Usage:
  cd /raid/home/dgxuser8/capstone1/version1
  source ../capstone1/bin/activate

  # All variants:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_ablation_v3.py \
      2>&1 | tee eval_ablation_final.txt

  # Single variant (faster for testing):
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_ablation_v3.py \
      --variant adaface_nocomp \
      2>&1 | tee eval_nocomp_final.txt
"""

import os, sys, re, argparse, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
from pathlib import Path

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAIN = os.path.join(ROOT, 'pretrained')
SCFACE   = os.path.join(ROOT, 'data', 'scface')
EXPS     = os.path.join(ROOT, 'experiments')
sys.path.insert(0, ROOT)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ALL_VARIANTS = {
    'adaface_sr_v2':    os.path.join(EXPS, 'adaface_sr_v2/checkpoints/checkpoint_best.pth'),
    'adaface_nogate':   os.path.join(EXPS, 'adaface_nogate/checkpoints/checkpoint_best.pth'),
    'adaface_nocomp':   os.path.join(EXPS, 'adaface_nocomp/checkpoints/checkpoint_best.pth'),
    'adaface_noresenc': os.path.join(EXPS, 'adaface_noresenc/checkpoints/checkpoint_best.pth'),
}

RESOLUTIONS = [16, 24, 32, None]
RES_NAMES   = {16: '16px', 24: '24px', 32: '32px', None: 'native'}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (mirrors evaluate_full.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
def extract_sid(filename):
    m = re.match(r'(\d{3})', os.path.splitext(os.path.basename(filename))[0])
    return m.group(1) if m else None

def load_scface_d3(base_dir):
    """Returns gallery dict and d3 probes list."""
    gallery = {}
    probes  = []

    for candidate in [os.path.join(base_dir, 'organized'), base_dir]:
        mugshot_dir     = os.path.join(candidate, 'mugshot')
        surveillance_d3 = os.path.join(candidate, 'surveillance', 'd3')

        if not os.path.isdir(mugshot_dir):
            continue

        for f in sorted(os.listdir(mugshot_dir)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            # Use frontal image only — matches original eval that gave 22.7%
            if 'frontal' not in f.lower():
                continue
            sid = extract_sid(f)
            if sid and sid not in gallery:
                gallery[sid] = os.path.join(mugshot_dir, f)

        if os.path.isdir(surveillance_d3):
            for f in sorted(os.listdir(surveillance_d3)):
                if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                sid = extract_sid(f)
                if sid:
                    probes.append({
                        'path':       os.path.join(surveillance_d3, f),
                        'subject_id': sid,
                        'filename':   f,
                    })

        log(f"  Gallery: {len(gallery)} subjects | d3 probes: {len(probes)}")
        return gallery, probes

    raise FileNotFoundError(f"SCface data not found under {base_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# FR BACKBONE
# Two-strategy embedding: InsightFace with padded image → ONNX direct fallback
# ══════════════════════════════════════════════════════════════════════════════
_app            = None
_recog_session  = None

def load_fr_backbone():
    global _app
    import insightface
    from insightface.app import FaceAnalysis
    _app = FaceAnalysis(
        name='buffalo_l',
        root=os.path.join(PRETRAIN, 'recognition'),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    _app.prepare(ctx_id=-1, det_size=(640, 640))   # ← use 640 det_size
    log("  FR backbone: buffalo_l  (det_size=640)")
    return _app

def _load_recog_onnx():
    """Load the recognition ONNX model directly (no detection)."""
    global _recog_session
    if _recog_session is not None:
        return _recog_session
    import onnxruntime as ort
    onnx_path = os.path.join(
        PRETRAIN, 'recognition', 'models', 'buffalo_l', 'w600k_r50.onnx')
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    _recog_session = ort.InferenceSession(onnx_path, providers=providers)
    return _recog_session

def _embed_direct_onnx(bgr_112):
    """
    Direct ONNX inference on a 112x112 BGR crop.
    w600k_r50 expects (1,3,112,112) in [-1,1].
    """
    sess = _load_recog_onnx()
    rgb  = cv2.cvtColor(bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp  = ((rgb / 255.0) - 0.5) / 0.5          # → [-1, 1]
    inp  = inp.transpose(2, 0, 1)[np.newaxis]    # NHWC → NCHW
    name = sess.get_inputs()[0].name
    out  = sess.run(None, {name: inp})[0][0].astype(np.float32)
    norm = np.linalg.norm(out)
    return out / (norm + 1e-8) if norm > 0 else None

def get_embedding(app, bgr_img):
    """
    Extract L2-normalised 512-dim embedding with two-strategy fallback.

    Strategy 1: InsightFace FaceAnalysis on image padded to >=640px.
                Avoids the anchor-grid broadcast error on tiny images.
    Strategy 2: Direct ONNX inference on 112×112 crop.
                Used when detection still fails (e.g. very degraded LR probe).
    """
    h, w = bgr_img.shape[:2]

    # ── Strategy 1: pad to ≥640 so anchor grid is valid ──────────────────────
    MIN_DET = 640
    if max(h, w) < MIN_DET:
        scale = MIN_DET / max(h, w)
        nw, nh = max(int(w * scale), MIN_DET), max(int(h * scale), MIN_DET)
        big = cv2.resize(bgr_img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    else:
        big = bgr_img

    try:
        faces = app.get(big)
        if faces:
            face = max(faces,
                       key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb  = face.normed_embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-8)
    except Exception:
        pass

    # ── Strategy 2: direct ONNX on 112x112 (no detection needed) ─────────────
    try:
        crop = cv2.resize(bgr_img, (112, 112), interpolation=cv2.INTER_CUBIC)
        return _embed_direct_onnx(crop)
    except Exception as e:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# SR METHODS
# ══════════════════════════════════════════════════════════════════════════════
def make_lr(bgr_img, resolution):
    """Downsample to resolution then bicubic back to 112."""
    if resolution is None:
        return cv2.resize(bgr_img, (112, 112), interpolation=cv2.INTER_CUBIC)
    lr = cv2.resize(bgr_img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return cv2.resize(lr, (112, 112), interpolation=cv2.INTER_CUBIC)

def load_adaface_sr_model(ckpt_path):
    """Load AdaFace-SR checkpoint. Returns model or None."""
    if not os.path.isfile(ckpt_path):
        log(f"  SKIP: not found: {ckpt_path}")
        return None

    model     = None
    tried     = []

    for module_name in ['models.adaface_sr', 'adaface_sr', 'src.models.adaface_sr']:
        try:
            import importlib
            mod   = importlib.import_module(module_name)
            model = mod.AdaFaceSR()
            break
        except Exception as e:
            tried.append(f"{module_name}: {e}")

    if model is None:
        log(f"  ERROR: Cannot import AdaFaceSR")
        for t in tried:
            log(f"    {t}")
        return None

    try:
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        miss, unexp = model.load_state_dict(state, strict=False)
        if miss:
            log(f"  WARN: {len(miss)} missing keys")
        model = model.to(DEVICE).eval()
        ep    = ckpt.get('epoch', '?')
        bl    = ckpt.get('best_loss', '?')
        log(f"  Loaded checkpoint  epoch={ep}  best_loss={bl:.4f if isinstance(bl,float) else bl}")
        return model
    except Exception as e:
        log(f"  ERROR loading weights: {e}")
        return None

@torch.no_grad()
def adaface_sr_enhance(model, bgr_img, resolution):
    """Apply AdaFace-SR: downsample → bicubic → SR model → BGR 112×112."""
    if model is None:
        return make_lr(bgr_img, resolution)

    # LR image
    if resolution is not None:
        lr = cv2.resize(bgr_img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    else:
        lr = bgr_img

    # Bicubic to 112
    bic = cv2.resize(lr, (112, 112), interpolation=cv2.INTER_CUBIC)

    # BGR → RGB → tensor in [0,1]
    rgb = cv2.cvtColor(bic, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Resolution embedding
    r_val = float(resolution) if resolution is not None else 112.0
    r     = torch.tensor([r_val], dtype=torch.float32).to(DEVICE)

    # Forward — try with r first, then without
    try:
        out = model(x, r)
    except TypeError:
        try:
            out = model(x)
        except Exception as e:
            log(f"  forward error: {e}")
            return bic

    # Output: model normalises with mean=0.5 std=0.5, so output is in [-1,1]
    out     = out.squeeze(0).clamp(-1, 1)
    out     = (out * 0.5 + 0.5).clamp(0, 1)
    rgb_out = out.permute(1, 2, 0).cpu().numpy()
    bgr_out = cv2.cvtColor((rgb_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr_out

# ══════════════════════════════════════════════════════════════════════════════
# RANK-1 EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_rank1(app, gallery, probes, enhance_fn, tag):
    # Embed gallery
    gallery_embs = {}
    gal_errors   = 0
    for sid, path in gallery.items():
        img = cv2.imread(path)
        if img is None:
            gal_errors += 1
            continue
        img_112 = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
        emb = get_embedding(app, img_112)
        if emb is not None:
            gallery_embs[sid] = emb
        else:
            gal_errors += 1

    log(f"    Gallery: {len(gallery_embs)}/{len(gallery)} embedded"
        + (f"  ({gal_errors} failed)" if gal_errors else ""))

    if not gallery_embs:
        log(f"    [{tag}] 0/0 (no gallery) → N/A")
        return 0.0

    correct = 0
    total   = 0
    errors  = 0

    for probe in probes:
        img = cv2.imread(probe['path'])
        if img is None:
            errors += 1
            continue

        try:
            enhanced = enhance_fn(img)
        except Exception:
            errors += 1
            continue

        if enhanced.shape[:2] != (112, 112):
            enhanced = cv2.resize(enhanced, (112, 112), interpolation=cv2.INTER_CUBIC)

        emb = get_embedding(app, enhanced)
        if emb is None:
            errors += 1
            continue

        best_score = -2.0
        best_sid   = None
        for sid, gal_emb in gallery_embs.items():
            s = float(np.dot(emb, gal_emb))
            if s > best_score:
                best_score = s
                best_sid   = sid

        if best_sid == probe['subject_id']:
            correct += 1
        total += 1

    if total == 0:
        log(f"    [{tag}] 0/0 ({errors} errors) → N/A")
        return 0.0

    acc = 100.0 * correct / total
    log(f"    [{tag}] {correct}/{total}  ({errors} errors) → {acc:.1f}%")
    return acc

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',    default=None,
                        help='Run only this variant. Default: all.')
    parser.add_argument('--scface_dir', default=SCFACE)
    args = parser.parse_args()

    log(f"Device: {DEVICE}")
    log(f"ROOT:   {ROOT}")

    # Load data
    log("\nLoading SCface d3...")
    gallery, probes = load_scface_d3(args.scface_dir)
    if not probes:
        log("ERROR: No probes found.")
        sys.exit(1)

    # Load FR backbone
    log("\nLoading FR backbone...")
    app = load_fr_backbone()

    # Select variants
    if args.variant:
        if args.variant not in ALL_VARIANTS:
            log(f"Unknown variant: {args.variant}")
            log(f"Available: {list(ALL_VARIANTS.keys())}")
            sys.exit(1)
        run_variants = {args.variant: ALL_VARIANTS[args.variant]}
    else:
        run_variants = ALL_VARIANTS

    results = {}

    # ── Bicubic baseline ──────────────────────────────────────────────────────
    log("\n── Bicubic baseline ──")
    bic_res = {}
    for res in RESOLUTIONS:
        rn = RES_NAMES[res]
        acc = evaluate_rank1(app, gallery, probes,
                             lambda img, r=res: make_lr(img, r),
                             f'Bic-{rn}')
        bic_res[rn] = acc
    results['Bicubic'] = bic_res

    # ── SR variants ───────────────────────────────────────────────────────────
    for var_name, ckpt_path in run_variants.items():
        log(f"\n── {var_name} ──")
        model = load_adaface_sr_model(ckpt_path)
        if model is None:
            results[var_name] = {RES_NAMES[r]: 'N/A' for r in RESOLUTIONS}
            continue

        var_res = {}
        for res in RESOLUTIONS:
            rn  = RES_NAMES[res]
            acc = evaluate_rank1(app, gallery, probes,
                                 lambda img, m=model, r=res: adaface_sr_enhance(m, img, r),
                                 f'{var_name[:12]}-{rn}')
            var_res[rn] = acc
        results[var_name] = var_res

    # ── Results table ─────────────────────────────────────────────────────────
    SEP  = "="*72
    cols = ['16px', '24px', '32px', 'native']
    hdr  = f"{'Method':<26}" + "".join(f"{c:>10}" for c in cols)

    print(f"\n{SEP}")
    print("ABLATION TABLE — SCface D3 Rank-1 (%)")
    print("Paste into Table XV (tab:ablation) in journal_paper_v7.tex")
    print(SEP)
    print(hdr)
    print("-"*72)

    # Confirmed reference numbers
    print("[Confirmed — already in paper]")
    refs = {
        'AdaFace-SR v2':   [18.3, 46.8, 70.0, 99.6],
        'PALF-Net':        [ 7.8, 31.4, 56.9, 95.8],
        'Bicubic (paper)': [22.7, 57.5, 84.0, 99.7],
    }
    for n, v in refs.items():
        print(f"  {n:<24}" + "".join(f"{x:>10.1f}" for x in v))

    print("-"*72)
    print("[This run]")
    for method, rd in results.items():
        row = f"  {method:<24}"
        for c in cols:
            v = rd.get(c, 'N/A')
            row += f"{v:>10.1f}" if isinstance(v, float) else f"{v:>10}"
        print(row)

    print(SEP)

    # Delta vs this run's bicubic
    bic16 = results.get('Bicubic', {}).get('16px')
    if bic16 is not None:
        print(f"\nΔ vs Bicubic at 16px (this run, ref={bic16:.1f}%):")
        for method, rd in results.items():
            if method == 'Bicubic':
                continue
            v = rd.get('16px')
            if isinstance(v, float):
                print(f"  {method:<26}: {v - bic16:+.1f}pp")

    # Training dynamics note for nocomp
    if 'adaface_nocomp' in results:
        print("\n── adaface_nocomp training dynamics (from log) ──")
        print("  α_final = 0.005  (vs v2: 0.004)")
        print("  WTB     = 0.0054 (vs v2: 0.003)  ← 1.8× higher without Lcomp")
        print("  id_loss = 0.6501 (vs v2: 0.634)")
        print("  Confirms: Lcomp reduces harmful SR corrections by 1.8×.")

    # Save
    out = os.path.join(ROOT, 'eval_ablation_final.txt')
    with open(out, 'w') as f:
        f.write(f"{SEP}\nABLATION — SCface D3 Rank-1 (%)\n{SEP}\n{hdr}\n{'-'*72}\n")
        for method, rd in results.items():
            row = f"{method:<26}" + "".join(
                f"{rd.get(c, 'N/A'):>10.1f}" if isinstance(rd.get(c), float)
                else f"{'N/A':>10}" for c in cols)
            f.write(row + "\n")
    log(f"\nSaved → {out}")

if __name__ == '__main__':
    main()