"""
generate_qualitative_figures.py  (v3 — syntax + path fixes)
------------------------------------------------------------
Run from project root:
    python scripts/generate_qualitative_figures.py \
        --adaface_ckpt experiments/adaface_sr_v2_clean/checkpoints/checkpoint_best.pth \
        --palfnet_ckpt experiments/palfnet_v1/checkpoints/checkpoint_best.pth \
        --scface_root  data/scface \
        --output_dir   results/figures \
        --n_candidates 300
"""

import argparse, sys, os
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.adaface_sr import AdaFaceSR
from src.models.palfnet import PALFNet

import insightface
from insightface.app import FaceAnalysis

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not installed. pip install scikit-image for PSNR.")

# ── threshold (module-level constant, no global needed) ──────────────────────
MATCH_THRESHOLD = 0.28


# ── path helpers ─────────────────────────────────────────────────────────────

def find_scface_dirs(root_str):
    """
    Returns (gallery_dir, d3_dir).
    Confirmed layout:
      gallery : data/scface/mugshot/
      d3 probes: data/scface/organized/surveillance/d3/
    """
    root = Path(root_str)
    layouts = [
        # Confirmed
        (root / "mugshot",
         root / "organized" / "surveillance" / "d3"),
        # Fallbacks
        (root / "mugshots",
         root / "organized" / "surveillance" / "d3"),
        (root / "mugshot",
         root / "surveillance" / "d3"),
        (root / "mugshots",
         root / "surveillance_cameras_distance_3"),
        (root / "SCface_database" / "mugshots",
         root / "SCface_database" / "surveillance_cameras_distance_3"),
    ]
    # Broad rglob fallback
    mugshot_dirs = [p for p in root.rglob("mugshot*") if p.is_dir()]
    d3_dirs      = [p for p in root.rglob("*")
                    if p.is_dir() and p.name == "d3"]
    for m in mugshot_dirs:
        for d in d3_dirs:
            layouts.append((m, d))

    print(f"Searching SCface dirs under: {root}")
    for gdir, d3dir in layouts:
        g_jpgs  = list(gdir.glob("*.jpg"))  if gdir.exists()  else []
        d3_jpgs = list(d3dir.glob("*.jpg")) if d3dir.exists() else []
        if g_jpgs and d3_jpgs:
            print(f"  Gallery : {gdir} ({len(g_jpgs)} jpgs)")
            print(f"  D3 probes: {d3dir} ({len(d3_jpgs)} jpgs)")
            return gdir, d3dir

    raise FileNotFoundError(
        f"Cannot locate SCface mugshots + d3 probes under {root}.\n"
        f"Run:  find {root} -name '*.jpg' | head -5\n"
        f"Then set --scface_root to the correct parent directory."
    )


# ── image utilities ───────────────────────────────────────────────────────────

def bgr2rgb(img): return img[:, :, ::-1].copy()

def t2bgr(t):
    img = t.squeeze(0).permute(1,2,0).cpu().float().numpy()
    return ((img*0.5+0.5).clip(0,1)*255).astype(np.uint8)[:,:,::-1]

def img2t(img_bgr, size, device):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size),
                     interpolation=cv2.INTER_AREA if size<64 else cv2.INTER_CUBIC)
    t = torch.from_numpy(rgb).float().permute(2,0,1)/255.0
    return ((t-0.5)/0.5).unsqueeze(0).to(device)

def get_emb(fa, img_bgr):
    """
    Get face embedding. Falls back to centre-crop embedding if detection fails,
    which is appropriate for aligned 112×112 SCface outputs.
    """
    try:
        faces = fa.get(img_bgr)
        if faces:
            f = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            return f.normed_embedding
    except Exception:
        pass
    # Fallback: use recognition model directly (image already 112×112 aligned)
    try:
        h, w = img_bgr.shape[:2]
        if h != 112 or w != 112:
            img_bgr = cv2.resize(img_bgr, (112,112), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Use insightface recognition model directly on aligned crop
        for model in fa.models.values():
            if hasattr(model, 'get_feat'):
                feat = model.get_feat(img_rgb)
                if feat is not None:
                    norm = np.linalg.norm(feat)
                    return feat / norm if norm > 0 else feat
    except Exception:
        pass
    return None

def cos(a, b):
    if a is None or b is None: return -1.0
    return float(np.dot(a, b))

def psnr(a_bgr, b_bgr):
    if not HAS_SKIMAGE: return 0.0
    a = cv2.resize(cv2.cvtColor(a_bgr,cv2.COLOR_BGR2RGB),(112,112)).astype(float)/255.
    b = cv2.resize(cv2.cvtColor(b_bgr,cv2.COLOR_BGR2RGB),(112,112)).astype(float)/255.
    return float(psnr_fn(a, b, data_range=1.0))


# ── inference ─────────────────────────────────────────────────────────────────

def run_ada(model, small_bgr, r, device):
    """
    AdaFaceSR.forward(x, lr_size):
      x       : (1,3,112,112) bicubic-upsampled in [-1,1]
      lr_size : (1,) scalar resolution
    Returns (output_bgr, alpha_mean, bic_bgr).
    """
    bic = cv2.resize(small_bgr, (112,112), interpolation=cv2.INTER_CUBIC)
    # [-1,1] normalisation matching training
    rgb = cv2.cvtColor(bic, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)
    t = (t - 0.5) / 0.5          # [-1, 1]
    lr_sz = torch.tensor([float(r)], device=device)
    with torch.no_grad():
        out_t, alpha_t = model(t, lr_sz)
    # Denorm [-1,1] → [0,255] BGR
    out_np = out_t.squeeze(0).permute(1,2,0).cpu().float().numpy()
    out_bgr = ((out_np * 0.5 + 0.5).clip(0,1) * 255).astype(np.uint8)[:,:,::-1].copy()
    return out_bgr, float(alpha_t.mean()), bic

def run_palf(model, small_bgr, device):
    """
    PALFNet.forward(lr, pose) always needs a pose tensor.
    For the no-pose ablation we pass zeros (equivalent to frontal pose).
    The model was trained with pose; zeros give the 'no pose conditioning'
    equivalent at inference — consistent with how the eval logs were produced.
    """
    bic = cv2.resize(small_bgr, (112,112), interpolation=cv2.INTER_CUBIC)
    x = img2t(bic, 112, device)
    # PALFNet signature: forward(lr, pose) — pose shape (B, 3)
    pose_zero = torch.zeros(x.shape[0], 3, device=device)
    with torch.no_grad():
        out = model(x, pose_zero)
        if isinstance(out, (tuple, list)): out = out[0]
        if isinstance(out, dict):
            out = out.get("output", out.get("sr", next(iter(out.values()))))
    # PALFNet outputs [0,1] range (see PALFNetInference.restore clamp(0,1))
    # Convert to BGR uint8 directly without [-1,1] denorm
    out_np = out.squeeze(0).permute(1,2,0).cpu().float().numpy()
    out_np = out_np.clip(0, 1)
    out_bgr = (out_np * 255).astype(np.uint8)[:, :, ::-1].copy()
    return out_bgr


# ── load SCface ───────────────────────────────────────────────────────────────

def extract_subject_id(stem):
    """
    Robustly extract numeric subject ID from SCface filename stems.
    Handles patterns:
      001           -> "1"
      001_01        -> "1"    (mugshot with angle index)
      001_cam1_d3   -> "1"    (probe with camera and distance)
      0001          -> "1"
    Always returns zero-padded 3-digit string for consistent matching.
    """
    import re
    # Extract leading digits
    m = re.match(r'^(\d+)', stem)
    if m:
        return str(int(m.group(1)))   # strip leading zeros, e.g. "001" -> "1"
    return stem


def load_scface(root_str):
    gdir, d3dir = find_scface_dirs(root_str)

    def glob_imgs(d):
        """Case-insensitive jpg glob — Linux glob is case-sensitive."""
        return sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPG"))
                      + list(d.glob("*.jpeg")) + list(d.glob("*.JPEG")))

    # Gallery: prefer _frontal image per subject; fall back to first found.
    gallery_all = {}   # subject_id -> list of paths
    for p in glob_imgs(gdir):
        s = extract_subject_id(p.stem)
        gallery_all.setdefault(s, []).append(p)

    gallery = {}
    for s, paths in gallery_all.items():
        # Prefer frontal, otherwise take first
        frontal = [p for p in paths if "frontal" in p.stem.lower()]
        chosen  = frontal[0] if frontal else paths[0]
        img = cv2.imread(str(chosen))
        if img is not None:
            gallery[s] = img

    probes = []
    for p in glob_imgs(d3dir):
        s = extract_subject_id(p.stem)
        img = cv2.imread(str(p))
        if img is not None:
            probes.append({"subj": s, "img": img})

    # Diagnostic: show a few ID samples so we can verify matching
    g_ids = sorted(gallery.keys())[:5]
    p_ids = sorted(set(x["subj"] for x in probes))[:5]
    print(f"Loaded: {len(gallery)} gallery subjects, {len(probes)} d3 probes")
    print(f"  Sample gallery IDs : {g_ids}")
    print(f"  Sample probe IDs   : {p_ids}")
    overlap = set(gallery.keys()) & set(x["subj"] for x in probes)
    print(f"  Matching subjects  : {len(overlap)}")
    if len(overlap) == 0:
        print("  WARNING: No ID overlap! Check filename patterns with:")
        print(f"    ls {gdir} | head -5")
        print(f"    ls {d3dir} | head -5")

    return gallery, probes


# ── scoring ───────────────────────────────────────────────────────────────────

def score_all(probes, gallery, fa, ada_model, palf_model, device, n):
    g_embs = {s: get_emb(fa, img) for s,img in gallery.items()}
    g_embs = {s:e for s,e in g_embs.items() if e is not None}
    print(f"  Gallery embeddings: {len(g_embs)}/130 extracted")

    out = []
    skipped_emb = 0
    for i, p in enumerate(probes[:n]):
        s = p["subj"]
        if s not in g_embs:
            skipped_emb += 1
            continue
        probe, hr = p["img"], gallery[s]
        small = cv2.resize(probe,(16,16),interpolation=cv2.INTER_AREA)
        bic   = cv2.resize(small,(112,112),interpolation=cv2.INTER_CUBIC)
        ada, alpha_m, _ = run_ada(ada_model, small, 16, device)
        palf = run_palf(palf_model, small, device) if palf_model else bic.copy()
        hr112 = cv2.resize(hr,(112,112),interpolation=cv2.INTER_AREA)
        ref   = g_embs[s]

        e_bic  = get_emb(fa, bic)
        e_ada  = get_emb(fa, ada)
        e_palf = get_emb(fa, palf)

        # Debug first probe
        if i == 0:
            print(f"  Probe 0 debug: subj={s} "
                  f"e_bic={'OK' if e_bic is not None else 'NONE'} "
                  f"e_ada={'OK' if e_ada is not None else 'NONE'} "
                  f"e_palf={'OK' if e_palf is not None else 'NONE'}")

        out.append({
            "subj":s, "probe":probe, "hr":hr,
            "small":small, "bic":bic, "ada":ada, "palf":palf,
            "s_bic": cos(e_bic,  ref),
            "s_ada": cos(e_ada,  ref),
            "s_palf":cos(e_palf, ref),
            "p_bic": psnr(bic,  hr112),
            "p_palf":psnr(palf, hr112),
            "p_ada": psnr(ada,  hr112),
            "alpha_mean": alpha_m,
        })
        if (i+1)%50==0: print(f"  {i+1}/{min(n,len(probes))} scored", flush=True)

    print(f"Scored {len(out)} probes. Skipped {skipped_emb} (no gallery emb).")
    return out


def select(scored, thr):
    rows = {}
    def best(cands, key): return max(cands, key=key) if cands else None

    # Fig A
    c = [x for x in scored if x["s_ada"]>thr and x["s_palf"]<thr]
    rows["ada_wins_palf_fails"] = best(c, lambda x: x["s_ada"]-x["s_palf"])

    c = [x for x in scored if x["s_ada"]>thr and x["s_bic"]<thr and x["s_palf"]<thr]
    if not c: c = [x for x in scored if x["s_ada"]>x["s_bic"] and x["s_ada"]>x["s_palf"]]
    rows["ada_only_win"] = best(c, lambda x: x["s_ada"])

    c = [x for x in scored if x["s_bic"]>thr and x["s_ada"]<thr]
    if not c: c = [x for x in scored if x["s_bic"]>x["s_ada"]+0.03]
    rows["honest_failure"] = best(c, lambda x: x["s_bic"]-x["s_ada"])

    # Fig B
    c = [x for x in scored if max(x["s_bic"],x["s_ada"],x["s_palf"])<thr]
    rows["all_fail"] = (min(c, key=lambda x: max(x["s_bic"],x["s_ada"]))
                        if c else None)

    c = [x for x in scored if x["p_palf"]>x["p_bic"] and x["s_palf"]<x["s_bic"]]
    rows["psnr_mismatch"] = best(c,
        lambda x:(x["p_palf"]-x["p_bic"])+(x["s_bic"]-x["s_palf"]))

    c = [x for x in scored if x["s_palf"]<x["s_bic"]-0.05 and x["s_palf"]<x["s_ada"]-0.03]
    if not c: c = scored
    rows["hallucination"] = min(c, key=lambda x: x["s_palf"]) if c else None

    return {k:v for k,v in rows.items() if v is not None}


# ── drawing ───────────────────────────────────────────────────────────────────

def score_color(s): return "#22aa44" if s >= MATCH_THRESHOLD else "#cc2222"

def draw_fig(rows_data, out_path):
    n = len(rows_data)
    if n == 0:
        print(f"Warning: no rows to draw for {out_path}")
        return
    col_titles = ["LR Input\n(16\u2009px)", "Bicubic",
                  "PALF-Net", "AdaFace-SR", "HR Reference"]
    fig, axes = plt.subplots(n, 5, figsize=(10.0, 2.5*n+0.5),
                             gridspec_kw={"wspace":0.04,"hspace":0.22})
    if n == 1: axes = axes[np.newaxis, :]
    for col, t in enumerate(col_titles):
        axes[0,col].set_title(t, fontsize=8.5, fontweight="bold", pad=2)

    for row, d in enumerate(rows_data):
        # LR: show native probe scaled up with NEAREST (pixelated — honest)
        lr_disp = cv2.resize(d["small"], (112,112), interpolation=cv2.INTER_NEAREST)
        hr_disp = cv2.resize(d["hr"],   (112,112),interpolation=cv2.INTER_AREA)
        imgs = [bgr2rgb(lr_disp), bgr2rgb(d["bic"]),
                bgr2rgb(d["palf"]), bgr2rgb(d["ada"]), bgr2rgb(hr_disp)]
        for col, img in enumerate(imgs):
            ax = axes[row,col]
            ax.imshow(img, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)

        # Cosine scores
        for col_i, sc in [(1,d["s_bic"]),(2,d["s_palf"]),(3,d["s_ada"])]:
            axes[row,col_i].text(0.5,-0.04, f"s={sc:.3f}",
                transform=axes[row,col_i].transAxes,
                ha="center", va="top", fontsize=7.5,
                color=score_color(sc), fontweight="bold")

        axes[row,0].set_ylabel(d.get("label",""), fontsize=8,
                               rotation=90, labelpad=2, va="center")

        # PSNR overlay (row B2)
        if d.get("show_psnr") and HAS_SKIMAGE:
            for col_i, (pv, tag) in enumerate(
                    [(d["p_bic"],""), (d["p_palf"],"↑"), (d["p_ada"],"")], 1):
                axes[row,col_i].text(0.5,1.03, f"PSNR={pv:.1f}dB{tag}",
                    transform=axes[row,col_i].transAxes,
                    ha="center", fontsize=7,
                    color="#cc6600" if col_i==2 else "#555555")

        # Red box (row B3) — eye region in 112×112 image coordinates
        if d.get("show_box"):
            for col_i in [2, 4]:
                axes[row,col_i].add_patch(
                    mpatches.Rectangle((23,28), 66,28,
                                       lw=1.8, ec="red", fc="none"))

    fig.text(0.5, 0.00,
             f"Threshold s={MATCH_THRESHOLD:.2f}:  "
             r"$\mathbf{green}$=match  $\mathbf{red}$=non-match",
             ha="center", fontsize=7.5, color="#444444")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(str(out_path)+ext, dpi=250, bbox_inches="tight")
        print(f"Saved: {out_path}{ext}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaface_ckpt", default=
        "experiments/adaface_sr_v2_clean/checkpoints/checkpoint_best.pth")
    parser.add_argument("--palfnet_ckpt", default=
        "experiments/palfnet_v1/checkpoints/checkpoint_best.pth")
    parser.add_argument("--scface_root",  default="data/scface")
    parser.add_argument("--output_dir",   default="results/figures")
    parser.add_argument("--n_candidates", type=int, default=300)
    parser.add_argument("--threshold",    type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    # Update module threshold from arg (no global needed — just reassign)
    thr = args.threshold

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fa = FaceAnalysis(name="buffalo_l", root="pretrained/recognition",
                      providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    fa.prepare(ctx_id=-1, det_size=(640, 640))

    print(f"Loading AdaFace-SR: {args.adaface_ckpt}")
    ada = AdaFaceSR().to(device); ada.eval()
    ck = torch.load(args.adaface_ckpt, map_location=device)
    ada.load_state_dict(ck.get("model_state_dict",ck.get("state_dict",ck)),
                        strict=False)

    palf_model = None
    try:
        print(f"Loading PALF-Net: {args.palfnet_ckpt}")
        # PALFNet.__init__ only accepts config=None — no use_pose arg
        palf_model = PALFNet().to(device); palf_model.eval()
        ck2 = torch.load(args.palfnet_ckpt, map_location=device)
        palf_model.load_state_dict(
            ck2.get("model_state_dict", ck2.get("state_dict", ck2)),
            strict=False)
        print("  PALF-Net loaded OK")
    except Exception as e:
        print(f"PALF-Net load failed ({e}). Using bicubic as PALF proxy.")

    gallery, probes = load_scface(args.scface_root)
    scored = score_all(probes, gallery, fa, ada, palf_model,
                       device, args.n_candidates)
    rows = select(scored, thr)
    print(f"Selected: {list(rows.keys())}")

    # Figure A
    fig_a = []
    for key, label in [
        ("ada_wins_palf_fails", "AdaFace-SR wins\nPALF-Net fails"),
        ("ada_only_win",        "AdaFace-SR only\nsuccess"),
        ("honest_failure",      "Honest failure\n(Bicubic wins)"),
    ]:
        if key in rows:
            d = dict(rows[key]); d["label"] = label; fig_a.append(d)
    draw_fig(fig_a, os.path.join(args.output_dir, "qualitative_comparison"))

    # Figure B
    fig_b = []
    for key, label, extras in [
        ("all_fail",      "Extreme pose\n(all fail)",    {}),
        ("psnr_mismatch", "PSNR–cosine\nmismatch",       {"show_psnr":True}),
        ("hallucination", "Identity\nhallucination",     {"show_box":True}),
    ]:
        if key in rows:
            d = dict(rows[key]); d["label"] = label; d.update(extras)
            fig_b.append(d)
    draw_fig(fig_b, os.path.join(args.output_dir, "failure_cases"))

    # Summary
    print("\n" + "="*60)
    print("PROBE SELECTION SUMMARY:")
    summary_lines = []
    for k, d in rows.items():
        line = (f"{k:30s}: subj={d['subj']}  "
                f"s_bic={d['s_bic']:.3f}  "
                f"s_palf={d['s_palf']:.3f}  "
                f"s_ada={d['s_ada']:.3f}")
        print(f"  {line}")
        summary_lines.append(line)
    print("="*60)

    sp = os.path.join(args.output_dir, "probe_selection_summary.txt")
    Path(sp).parent.mkdir(parents=True, exist_ok=True)
    with open(sp,"w") as f: f.write("\n".join(summary_lines))
    print(f"Saved: {sp}")


if __name__ == "__main__":
    main()