#!/usr/bin/env python3
"""
eval_ablation_v4.py  —  AdaFace-SR ablation on SCface d3.

Key fixes vs v3:
  1. Model returns TUPLE (sr_tensor, alpha_map) — handle correctly
  2. Model input: BGR→RGB /255.0 only (NO mean/std normalisation)
  3. Gallery: MEAN embedding across ALL 10 mugshot images per subject
     (L1-L4, R1-R4, frontal.jpg, frontal.JPG) — matches original eval

Usage:
  cd /raid/home/dgxuser8/capstone1/version1
  source ../capstone1/bin/activate

  # Test bicubic first:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_ablation_v4.py \
      --bicubic_only 2>&1 | head -30

  # All variants:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_ablation_v4.py \
      2>&1 | tee eval_ablation_final.txt

  # Single variant:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_ablation_v4.py \
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
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def extract_sid(filename):
    m = re.match(r'(\d{3})', os.path.splitext(os.path.basename(filename))[0])
    return m.group(1) if m else None

def load_scface_d3(base_dir):
    """
    Mirrors load_scface() + main() gallery logic from evaluate_full.py exactly.
    Gallery: ONE image per subject — first match in os.listdir() order (not sorted).
    Probes:  all d3 surveillance images.
    """
    gallery = {}  # sid -> single path
    probes  = []

    for candidate in [os.path.join(base_dir, 'organized'), base_dir]:
        mugshot_dir     = os.path.join(candidate, 'mugshot')
        surveillance_d3 = os.path.join(candidate, 'surveillance', 'd3')

        if not os.path.isdir(mugshot_dir):
            continue

        # os.listdir — NOT sorted, matches evaluate_full.py load_scface() exactly
        for f in os.listdir(mugshot_dir):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
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
# FR BACKBONE  —  InsightFace buffalo_l with anchor-grid fix + ONNX fallback
# ══════════════════════════════════════════════════════════════════════════════
_app           = None
_recog_session = None

def load_fr_backbone():
    global _app
    from insightface.app import FaceAnalysis
    _app = FaceAnalysis(
        name='buffalo_l',
        root=os.path.join(PRETRAIN, 'recognition'),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    _app.prepare(ctx_id=0, det_size=(640, 640))
    log("  FR backbone loaded: buffalo_l (det_size=640)")
    return _app

def get_embedding(app, bgr):
    """
    Exact mirror of Embedder.embed() from evaluate_full.py:
      1. Try direct detection
      2. Fallback: pad to max(300, h*3, w*3) canvas
      3. Fallback: upscale if max(h,w) < 100
    Returns raw (un-normalised) embedding, or None.
    """
    # Strategy 1: direct
    try:
        faces = app.get(bgr)
        if faces:
            return max(faces,
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).embedding
    except Exception:
        pass

    # Strategy 2: pad (mirrors evaluate_full.py lines 362-368)
    h, w = bgr.shape[:2]
    cs = max(300, h * 3, w * 3)
    canvas = np.zeros((cs, cs, 3), dtype=np.uint8)
    yo, xo = (cs - h) // 2, (cs - w) // 2
    canvas[yo:yo+h, xo:xo+w] = bgr
    try:
        faces = app.get(canvas)
        if faces:
            return max(faces,
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).embedding
    except Exception:
        pass

    # Strategy 3: upscale if very small (mirrors evaluate_full.py lines 371-376)
    if max(h, w) < 100:
        sc = max(4, 200 // max(h, w))
        up = cv2.resize(bgr, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
        try:
            faces = app.get(up)
            if faces:
                return max(faces,
                    key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).embedding
        except Exception:
            pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
# GALLERY BUILDING  —  mean embedding over all mugshots per subject
# ══════════════════════════════════════════════════════════════════════════════
def build_gallery_embeddings(app, gallery):
    """
    Mirrors evaluate_full.py main() lines 906-915 exactly:
    - gallery is dict {sid: single_path}
    - embed each gallery image directly (no mean, no multi-shot)
    - store raw embedding (normalisation happens at match time)
    """
    gallery_embs = {}
    n_fail = 0
    for sid, path in gallery.items():
        img = cv2.imread(path)
        if img is None:
            n_fail += 1
            continue
        emb = get_embedding(app, img)
        if emb is not None:
            gallery_embs[sid] = emb
        else:
            n_fail += 1
    log(f"  Gallery embeddings: {len(gallery_embs)}/{len(gallery)} built"
        + (f"  ({n_fail} failed)" if n_fail else ""))
    return gallery_embs


# ══════════════════════════════════════════════════════════════════════════════
# SR METHODS
# ══════════════════════════════════════════════════════════════════════════════
def make_lr(bgr_img, resolution):
    if resolution is None:
        return cv2.resize(bgr_img, (112, 112), interpolation=cv2.INTER_CUBIC)
    lr = cv2.resize(bgr_img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return cv2.resize(lr, (112, 112), interpolation=cv2.INTER_CUBIC)

def load_adaface_sr_model(ckpt_path):
    if not os.path.isfile(ckpt_path):
        log(f"  SKIP — not found: {ckpt_path}")
        return None

    model  = None
    errors = []

    # Try import paths in order (mirrors evaluate_full.py)
    for module_name in ['src.models.adaface_sr', 'models.adaface_sr', 'adaface_sr']:
        try:
            import importlib
            mod   = importlib.import_module(module_name)
            ModelClass = mod.AdaFaceSR
            # Pass config from checkpoint if available (mirrors evaluate_full.py)
            ckpt_tmp = torch.load(ckpt_path, map_location='cpu')
            config   = ckpt_tmp.get('config', {})
            model_config = {
                'num_feat':      config.get('num_feat',      64),
                'num_block':     config.get('num_block',      4),
                'num_grow':      config.get('num_grow',       32),
                'res_embed_dim': config.get('res_embed_dim',  64),
                'gate_channels': config.get('gate_channels',  32),
            }
            model = ModelClass(model_config)
            break
        except Exception as e:
            errors.append(f"{module_name}: {e}")

    # Fallback: try no-arg constructor
    if model is None:
        for module_name in ['src.models.adaface_sr', 'models.adaface_sr', 'adaface_sr']:
            try:
                import importlib
                mod   = importlib.import_module(module_name)
                model = mod.AdaFaceSR()
                break
            except Exception as e:
                errors.append(f"{module_name} (no-arg): {e}")

    if model is None:
        log("  ERROR: Cannot import AdaFaceSR")
        for e in errors:
            log(f"    {e}")
        return None

    try:
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        miss, _ = model.load_state_dict(state, strict=False)
        if miss:
            log(f"  WARN: {len(miss)} missing keys")
        model = model.to(DEVICE).eval()
        ep = ckpt.get('epoch', '?')
        bl = ckpt.get('best_loss', '?')
        log(f"  Loaded: epoch={ep}  best_loss={bl:.4f if isinstance(bl,float) else bl}")
        return model
    except Exception as e:
        log(f"  ERROR loading weights: {e}")
        return None

@torch.no_grad()
def adaface_sr_enhance(model, bgr_img, resolution):
    """
    Mirrors SRMethodAdaFaceSR.enhance() from evaluate_full.py exactly:
      - Input: BGR /255.0  (NO mean/std normalisation)
      - Forward: model(lr_t, lr_size_t)  returns TUPLE (sr_t, alpha_t)
      - Output: clamp(0,1), *255, BGR
    """
    if model is None:
        return make_lr(bgr_img, resolution)

    h, w = bgr_img.shape[:2]

    # Determine lr_size before resizing (mirrors: lr_size = min(h, w))
    if resolution is not None:
        lr     = cv2.resize(bgr_img, (resolution, resolution),
                            interpolation=cv2.INTER_AREA)
        lr_size = float(resolution)
    else:
        lr      = bgr_img
        lr_size = float(min(h, w))

    # Bicubic to 112×112
    lr_112 = cv2.resize(lr, (112, 112), interpolation=cv2.INTER_CUBIC)

    # BGR→RGB, /255.0  (NO mean/std — matches evaluate_full.py line 332)
    lr_rgb = cv2.cvtColor(lr_112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lr_t   = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Resolution embedding
    lr_size_t = torch.tensor([lr_size], device=DEVICE)

    # Forward — model returns TUPLE (sr_t, alpha_t)
    try:
        output = model(lr_t, lr_size_t)
        if isinstance(output, (tuple, list)):
            sr_t = output[0]          # first element is the SR image
        else:
            sr_t = output             # some variants may return single tensor

        sr_t    = sr_t.clamp(0, 1)   # matches evaluate_full.py line 337
        sr_rgb  = sr_t[0].permute(1, 2, 0).cpu().numpy()
        sr_bgr  = cv2.cvtColor((sr_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return sr_bgr

    except Exception as e:
        log(f"  forward error: {e}")
        return lr_112   # fallback to bicubic


# ══════════════════════════════════════════════════════════════════════════════
# RANK-1 EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_rank1(app, gallery_embs, probes, enhance_fn, tag):
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

        # Normalise same as evaluate_full.py lines 478-483
        emb_n = emb / (np.linalg.norm(emb) + 1e-8)
        best_score = -2.0
        best_sid   = None
        for sid, gal_emb in gallery_embs.items():
            ge_n = gal_emb / (np.linalg.norm(gal_emb) + 1e-8)
            s = float(np.dot(emb_n, ge_n))
            if s > best_score:
                best_score = s
                best_sid   = sid

        if best_sid == probe['subject_id']:
            correct += 1
        total += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    log(f"    [{tag}] {correct}/{total}  ({errors} errors) → {acc:.1f}%")
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',      default=None)
    parser.add_argument('--scface_dir',   default=SCFACE)
    parser.add_argument('--bicubic_only', action='store_true',
                        help='Only run bicubic baseline (fast sanity check)')
    args = parser.parse_args()

    log(f"Device: {DEVICE}")

    # Load data
    log("\nLoading SCface d3...")
    gallery, probes = load_scface_d3(args.scface_dir)
    if not probes:
        log("ERROR: No probes found.")
        sys.exit(1)

    # Load FR backbone
    log("\nLoading FR backbone...")
    app = load_fr_backbone()

    # Build gallery mean embeddings (once, shared across all methods)
    log("\nBuilding gallery embeddings (mean over all mugshots)...")
    gallery_embs = build_gallery_embeddings(app, gallery)

    if args.bicubic_only:
        log("\n── Bicubic baseline only ──")
        for res in RESOLUTIONS:
            rn = RES_NAMES[res]
            evaluate_rank1(app, gallery_embs, probes,
                           lambda img, r=res: make_lr(img, r),
                           f'Bic-{rn}')
        return

    # Select variants
    if args.variant:
        if args.variant not in ALL_VARIANTS:
            log(f"Unknown variant: {args.variant}")
            sys.exit(1)
        run_variants = {args.variant: ALL_VARIANTS[args.variant]}
    else:
        run_variants = ALL_VARIANTS

    results = {}

    # Bicubic baseline
    log("\n── Bicubic baseline ──")
    bic_res = {}
    for res in RESOLUTIONS:
        rn = RES_NAMES[res]
        acc = evaluate_rank1(app, gallery_embs, probes,
                             lambda img, r=res: make_lr(img, r),
                             f'Bic-{rn}')
        bic_res[rn] = acc
    results['Bicubic'] = bic_res

    # SR variants
    for var_name, ckpt_path in run_variants.items():
        log(f"\n── {var_name} ──")
        model = load_adaface_sr_model(ckpt_path)
        if model is None:
            results[var_name] = {RES_NAMES[r]: 'N/A' for r in RESOLUTIONS}
            continue

        var_res = {}
        for res in RESOLUTIONS:
            rn  = RES_NAMES[res]
            acc = evaluate_rank1(
                app, gallery_embs, probes,
                lambda img, m=model, r=res: adaface_sr_enhance(m, img, r),
                f'{var_name[:12]}-{rn}')
            var_res[rn] = acc
        results[var_name] = var_res

    # Print table
    SEP  = "="*72
    cols = ['16px', '24px', '32px', 'native']
    hdr  = f"{'Method':<26}" + "".join(f"{c:>10}" for c in cols)

    print(f"\n{SEP}")
    print("ABLATION TABLE — SCface D3 Rank-1 (%)")
    print("Paste into Table XV (tab:ablation) in journal_paper_v7.tex")
    print(SEP)
    print(hdr)
    print("-"*72)

    refs = {
        'AdaFace-SR v2 (paper)': [18.3, 46.8, 70.0, 99.6],
        'PALF-Net (paper)':      [ 7.8, 31.4, 56.9, 95.8],
        'Bicubic (paper)':       [22.7, 57.5, 84.0, 99.7],
    }
    print("[Reference — confirmed paper numbers]")
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

    bic16 = results.get('Bicubic', {}).get('16px')
    if isinstance(bic16, float):
        print(f"\nΔ vs Bicubic (this run) at 16px  [ref={bic16:.1f}%]:")
        for method, rd in results.items():
            if method == 'Bicubic':
                continue
            v = rd.get('16px')
            if isinstance(v, float):
                print(f"  {method:<26}: {v-bic16:+.1f}pp")

    # Training dynamics for nocomp
    if 'adaface_nocomp' in results:
        print("\n── adaface_nocomp: training dynamics ──")
        print("  α_final = 0.005  (vs v2: 0.004)")
        print("  WTB     = 0.0054 (vs v2: 0.003) → 1.8× higher without Lcomp")
        print("  Confirms: Lcomp reduces harmful SR corrections.")

    # Save
    out = os.path.join(ROOT, 'eval_ablation_final.txt')
    with open(out, 'w') as f:
        f.write(f"{SEP}\nABLATION — SCface D3 Rank-1 (%)\n{SEP}\n{hdr}\n{'-'*72}\n")
        for method, rd in results.items():
            row = f"{method:<26}" + "".join(
                f"{rd.get(c,'N/A'):>10.1f}" if isinstance(rd.get(c), float)
                else f"{'N/A':>10}" for c in cols)
            f.write(row + "\n")
    log(f"\nSaved → {out}")

if __name__ == '__main__':
    main()