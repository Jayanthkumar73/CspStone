#!/usr/bin/env python3
"""
phase1_baseline_eval.py — Baseline FR evaluation with pose-stratified analysis.

This script proves (or disproves) the core hypothesis:
    "Pose hurts MORE at low resolution than at high resolution"

What it does:
  1. Loads SCface images at d1/d2/d3 + mugshots (gallery)
  2. Runs Real-ESRGAN + 6DRepNet to estimate pose for each probe
  3. Extracts face embeddings using InsightFace ArcFace R100
  4. Computes Rank-1/5 accuracy OVERALL and STRATIFIED by pose bin
  5. Produces the cross-tabulated results table (resolution x pose)
  6. Generates publication-ready plots

Usage:
    python version1/scripts/phase1_baseline_eval.py
    python version1/scripts/phase1_baseline_eval.py --scface_dir data/scface
"""

import os, sys, time, json, argparse, re, warnings
warnings.filterwarnings("ignore")
import sys
import torchvision

# MOCK FIX: Redirect the removed functional_tensor module to the new location
try:
    import torchvision.transforms.functional_tensor as T_rt
except ImportError:
    import torchvision.transforms.functional as T_rt
    sys.modules['torchvision.transforms.functional_tensor'] = T_rt
import numpy as np
import cv2
import torch
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results", "phase1")
SCFACE_ORG = os.path.join(ROOT, "data", "scface", "organized")
PRETRAINED = os.path.join(ROOT, "pretrained")

POSE_BINS = {
    "frontal":      (0,  15),
    "half_profile": (15, 45),
    "profile":      (45, 90),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════
def extract_sid(filename):
    m = re.match(r'(\d{3})', os.path.splitext(filename)[0])
    return m.group(1) if m else None


def load_scface(base_dir):
    """Load SCface — tries organized/ then raw scan."""
    gallery, probes = {}, {"d1": [], "d2": [], "d3": []}

    # Try organized structure
    org = os.path.join(base_dir, "organized")
    mug_dir = None
    surv_dir = None

    for candidate in [org, base_dir]:
        md = os.path.join(candidate, "mugshot")
        sd = os.path.join(candidate, "surveillance")
        if os.path.isdir(md):
            mug_dir = md
            surv_dir = sd
            break

    if mug_dir and os.path.isdir(mug_dir):
        for f in sorted(os.listdir(mug_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                sid = extract_sid(f)
                if sid:
                    gallery[sid] = os.path.join(mug_dir, f)

        if surv_dir:
            for dist in ["d1", "d2", "d3"]:
                dd = os.path.join(surv_dir, dist)
                if not os.path.isdir(dd):
                    continue
                for f in sorted(os.listdir(dd)):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sid = extract_sid(f)
                        if sid:
                            probes[dist].append({"path": os.path.join(dd, f),
                                                  "subject_id": sid, "filename": f})
        return gallery, probes

    # Fallback: raw scan
    log("  Organized structure not found, scanning raw directory...")
    for dp, _, fns in os.walk(base_dir):
        for f in fns:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            p = os.path.join(dp, f)
            sid = extract_sid(f)
            if not sid:
                continue
            pl = p.lower()
            if any(x in pl for x in ['mugshot', 'frontal', 'gallery', '/hr']):
                gallery[sid] = p
            elif any(x in pl for x in ['distance1', 'dist1', '/d1/', '_d1']):
                probes["d1"].append({"path": p, "subject_id": sid, "filename": f})
            elif any(x in pl for x in ['distance2', 'dist2', '/d2/', '_d2']):
                probes["d2"].append({"path": p, "subject_id": sid, "filename": f})
            elif any(x in pl for x in ['distance3', 'dist3', '/d3/', '_d3']):
                probes["d3"].append({"path": p, "subject_id": sid, "filename": f})

    return gallery, probes


# ═══════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════
class Embedder:
    """InsightFace ArcFace R100 embedding extractor."""

    def __init__(self):
        log("Loading InsightFace ArcFace R100...")
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=os.path.join(PRETRAINED, "recognition"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        log("  ✅ Ready")

    def embed(self, bgr, try_upscale=True):
        """Get 512-d embedding. Handles small images via padding/upscaling."""
        # Direct attempt
        faces = self.app.get(bgr)
        if faces:
            return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).embedding

        h, w = bgr.shape[:2]

        # Pad into larger canvas
        cs = max(300, h * 3, w * 3)
        canvas = np.zeros((cs, cs, 3), dtype=np.uint8)
        yo, xo = (cs - h) // 2, (cs - w) // 2
        canvas[yo:yo+h, xo:xo+w] = bgr
        faces = self.app.get(canvas)
        if faces:
            return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).embedding

        # Upscale
        if try_upscale and max(h, w) < 100:
            sc = max(4, 200 // max(h, w))
            up = cv2.resize(bgr, (w * sc, h * sc), interpolation=cv2.INTER_CUBIC)
            faces = self.app.get(up)
            if faces:
                return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).embedding

        return None


class PoseEst:
    """6DRepNet with Real-ESRGAN preprocessing for small faces."""

    def __init__(self):
        log("Loading pose pipeline...")
        from sixdrepnet import SixDRepNet
        self.model = SixDRepNet()

        self.sr = None
        rp = os.path.join(PRETRAINED, "sr", "RealESRGAN_x4plus.pth")
        if os.path.isfile(rp):
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)
                self.sr = RealESRGANer(scale=4, model_path=rp, model=net,
                                        tile=0, tile_pad=10, pre_pad=0, half=True)
                log("  ✅ Real-ESRGAN loaded (pose preprocessing)")
            except Exception as e:
                log(f"  ⚠️  Real-ESRGAN: {e}")
        log("  ✅ Pose pipeline ready")

    def estimate(self, bgr):
        h, w = bgr.shape[:2]
        img = bgr
        # Phase 0 findings:
        #   Real-ESRGAN helps at 16-48px (cuts error ~50%)
        #   Real-ESRGAN HURTS below 12px (49.7° vs 25.6° at 8×8)
        #   So only apply SR in the sweet spot: 12px <= face < 48px
        if 12 <= max(h, w) < 48 and self.sr:
            try:
                img, _ = self.sr.enhance(bgr, outscale=4)
            except:
                pass
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        try:
            y, p, r = self.model.predict(img)
            if isinstance(y, (list, np.ndarray)):
                y, p, r = float(y[0]), float(p[0]), float(r[0])
            return float(y), float(p), float(r)
        except:
            return None, None, None


def pose_bin(yaw):
    if yaw is None:
        return "unknown"
    a = abs(yaw)
    for name, (lo, hi) in POSE_BINS.items():
        if lo <= a < hi:
            return name
    return "profile"


# ═══════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════
def evaluate(gallery_embs, gallery_ids, probes, embedder, pose_est, dist_name):
    results = []
    n = len(probes)
    sizes = []

    for i, entry in enumerate(probes):
        img = cv2.imread(entry["path"])
        if img is None:
            continue

        h, w = img.shape[:2]
        sizes.append(max(h, w))

        emb = embedder.embed(img)
        if emb is None:
            results.append({"subject_id": entry["subject_id"],
                            "filename": entry["filename"], "detected": False,
                            "img_h": h, "img_w": w})
            continue

        yaw, pitch, roll = pose_est.estimate(img)

        # Cosine similarities to all gallery
        emb_n = emb / (np.linalg.norm(emb) + 1e-8)
        sims = {}
        for gid in gallery_ids:
            ge = gallery_embs[gid]
            ge_n = ge / (np.linalg.norm(ge) + 1e-8)
            sims[gid] = float(np.dot(emb_n, ge_n))

        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        rank = next((r + 1 for r, (gid, _) in enumerate(ranked)
                      if gid == entry["subject_id"]), -1)

        results.append({
            "subject_id": entry["subject_id"],
            "filename": entry["filename"],
            "detected": True,
            "img_h": h, "img_w": w,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "pose_bin": pose_bin(yaw),
            "rank": rank,
            "top1_id": ranked[0][0] if ranked else None,
            "top1_sim": ranked[0][1] if ranked else 0,
            "correct_sim": sims.get(entry["subject_id"], 0),
        })

        if (i + 1) % 100 == 0 or i == n - 1:
            det = sum(1 for r in results if r.get("detected"))
            log(f"    {dist_name}: {i+1}/{n}  detected={det}")

    if sizes:
        log(f"    {dist_name} image sizes: min={min(sizes)}px  max={max(sizes)}px  "
            f"mean={np.mean(sizes):.0f}px  median={np.median(sizes):.0f}px")

    return results


def metrics(results, ranks=[1, 5, 10]):
    det = [r for r in results if r.get("detected")]
    if not det:
        return {f"rank{n}": 0.0 for n in ranks} | {"n": 0, "n_det": 0}
    m = {}
    for n in ranks:
        m[f"rank{n}"] = sum(1 for r in det if 0 < r.get("rank", -1) <= n) / len(det)
    m["n"] = len(results)
    m["n_det"] = len(det)
    return m


def stratified(results):
    by_bin = defaultdict(list)
    for r in results:
        if r.get("detected") and r.get("pose_bin"):
            by_bin[r["pose_bin"]].append(r)
    out = {}
    for bn in ["frontal", "half_profile", "profile", "unknown"]:
        if bn in by_bin:
            out[bn] = metrics(by_bin[bn])
            out[bn]["count"] = len(by_bin[bn])
        else:
            out[bn] = {"rank1": 0, "rank5": 0, "count": 0}
    return out


# ═══════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════
def report(all_res, all_strat):
    L = []
    L.append("\n" + "=" * 80)
    L.append("  PHASE 1: SCface Baseline — Pose-Stratified Analysis")
    L.append("=" * 80)

    L.append("\n  ── Overall ──")
    L.append(f"  {'Dist':>6} | {'Rank-1':>8} | {'Rank-5':>8} | {'Rank-10':>8} | {'Det/Total':>12}")
    L.append("  " + "-" * 55)
    for d in ["d1", "d2", "d3"]:
        if d not in all_res:
            continue
        m = metrics(all_res[d])
        L.append(f"  {d:>6} | {m['rank1']:>7.1%} | {m.get('rank5',0):>7.1%} | "
                 f"{m.get('rank10',0):>7.1%} | {m['n_det']:>5}/{m['n']:<5}")

    L.append("\n  ══ POSE × RESOLUTION TABLE (Paper Table 1) ══")
    L.append(f"  {'Dist':>6} | {'Frontal':>12} | {'Half-Prof':>12} | {'Profile':>12} | {'Δ(F−P)':>8} | {'ALL':>8}")
    L.append("  " + "-" * 72)
    deltas = {}
    for d in ["d1", "d2", "d3"]:
        if d not in all_strat:
            continue
        s = all_strat[d]
        fr = s.get("frontal", {})
        hp = s.get("half_profile", {})
        pr = s.get("profile", {})
        a = metrics(all_res[d])
        f1 = fr.get("rank1", 0)
        p1 = pr.get("rank1", 0)
        delta = f1 - p1
        deltas[d] = delta
        L.append(
            f"  {d:>6} | {f1:>6.1%} (n={fr.get('count',0):>3}) | "
            f"{hp.get('rank1',0):>6.1%} (n={hp.get('count',0):>3}) | "
            f"{p1:>6.1%} (n={pr.get('count',0):>3}) | {delta:>+7.1%} | {a['rank1']:>7.1%}")

    L.append("\n  ── Hypothesis: Pose gap INCREASES at lower resolution ──")
    if "d1" in deltas and "d3" in deltas:
        if deltas["d1"] > deltas["d3"] + 0.02:
            L.append(f"  ✅ CONFIRMED  d1 Δ={deltas['d1']:+.1%}  >  d3 Δ={deltas['d3']:+.1%}")
            L.append("  → Pose hurts MORE at low resolution → PALF-Net is justified!")
        elif abs(deltas["d1"] - deltas["d3"]) <= 0.02:
            L.append(f"  ⚠️  INCONCLUSIVE  d1 Δ={deltas['d1']:+.1%}  ≈  d3 Δ={deltas['d3']:+.1%}")
            L.append("  → Pose gap similar — PALF-Net may still help on absolute numbers")
        else:
            L.append(f"  ❌ NOT CONFIRMED  d1 Δ={deltas['d1']:+.1%}  <  d3 Δ={deltas['d3']:+.1%}")
            L.append("  → Consider pivoting approach")
    elif deltas:
        L.append(f"  Only partial data: {deltas}")

    L.append("\n" + "=" * 80)
    return "\n".join(L)


def plots(all_res, all_strat):
    os.makedirs(os.path.join(RESULTS, "plots"), exist_ok=True)

    dists = [d for d in ["d1", "d2", "d3"] if d in all_strat]
    if not dists:
        return

    # ── Bar chart: pose bins per distance ──
    fig, axes = plt.subplots(1, len(dists), figsize=(5 * len(dists), 5), sharey=True)
    if len(dists) == 1:
        axes = [axes]
    bins = ["frontal", "half_profile", "profile"]
    labels = ["Frontal\n(0-15°)", "Half-Prof\n(15-45°)", "Profile\n(45-90°)"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    res_lbl = {"d1": "d1 (4.2m ≈15px)", "d2": "d2 (2.6m ≈30px)", "d3": "d3 (1.0m ≈50px)"}

    for ax, d in zip(axes, dists):
        s = all_strat[d]
        vals = [s.get(b, {}).get("rank1", 0) * 100 for b in bins]
        cnts = [s.get(b, {}).get("count", 0) for b in bins]
        bars = ax.bar(range(3), vals, color=colors, width=0.6, edgecolor="black", lw=0.5)
        for bar, v, c in zip(bars, vals, cnts):
            if c:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{v:.1f}%\nn={c}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(res_lbl.get(d, d), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=.3)
        if d == dists[0]:
            ax.set_ylabel("Rank-1 (%)", fontsize=11)

    plt.suptitle("Pose-Stratified Rank-1 — SCface Baseline (ArcFace R100)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "plots", "pose_stratified_rank1.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log("  Plot: pose_stratified_rank1.png")

    # ── Frontal vs Profile comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(dists))
    w = 0.3
    fv = [all_strat[d].get("frontal", {}).get("rank1", 0) * 100 for d in dists]
    pv = [all_strat[d].get("profile", {}).get("rank1", 0) * 100 for d in dists]
    ax.bar(x - w / 2, fv, w, label="Frontal (0-15°)", color="#2ecc71", edgecolor="black", lw=.5)
    ax.bar(x + w / 2, pv, w, label="Profile (45-90°)", color="#e74c3c", edgecolor="black", lw=.5)
    for i in range(len(dists)):
        gap = fv[i] - pv[i]
        ax.annotate(f"Δ={gap:+.1f}%", xy=(x[i] + 0.45, (fv[i] + pv[i]) / 2),
                    fontsize=9, fontweight="bold", color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels([res_lbl.get(d, d) for d in dists], fontsize=10)
    ax.set_ylabel("Rank-1 (%)", fontsize=11)
    ax.set_title("Pose Gap Across Distances", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "plots", "pose_gap.png"), dpi=200, bbox_inches="tight")
    plt.close()
    log("  Plot: pose_gap.png")

    # ── Yaw histograms ──
    fig, axes = plt.subplots(1, len(dists), figsize=(5 * len(dists), 4), sharey=True)
    if len(dists) == 1:
        axes = [axes]
    for ax, d in zip(axes, dists):
        yaws = [r["yaw"] for r in all_res[d] if r.get("detected") and r.get("yaw") is not None]
        if yaws:
            ax.hist(yaws, bins=30, range=(-90, 90), color="#3498db", alpha=.7, edgecolor="black", lw=.5)
            for v in [-45, -15, 15, 45]:
                ax.axvline(v, color="orange" if abs(v) == 15 else "red", ls=":", alpha=.5)
        ax.set_xlabel("Yaw (°)")
        ax.set_title(f"{d} (n={len(yaws)})")
        if d == dists[0]:
            ax.set_ylabel("Count")
    plt.suptitle("Yaw Distribution per Distance", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "plots", "yaw_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("  Plot: yaw_dist.png")


def save_json(all_res, all_strat):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    for d, res in all_res.items():
        with open(os.path.join(RESULTS, f"results_{d}.json"), "w") as f:
            json.dump([{k: conv(v) for k, v in r.items()} for r in res], f, indent=2, default=conv)

    summary = {}
    for d in ["d1", "d2", "d3"]:
        if d in all_res:
            summary[d] = {"overall": metrics(all_res[d]),
                          "stratified": {bn: {k: conv(v) for k, v in m.items()}
                                         for bn, m in all_strat.get(d, {}).items()}}
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=conv)
    log(f"  JSON saved to {RESULTS}/")


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scface_dir", default=os.path.join(ROOT, "data", "scface"))
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    log("=" * 70)
    log("  PALF-Net Phase 1: Baseline + Pose-Stratified Analysis")
    log("=" * 70)

    # Load data
    gallery, probes = load_scface(args.scface_dir)
    log(f"\n  Gallery: {len(gallery)} subjects")
    for d in ["d1", "d2", "d3"]:
        log(f"  Probes {d}: {len(probes[d])}")

    if not gallery:
        log("❌ No gallery. Run: python scripts/organize_scface.py")
        sys.exit(1)

    # Models
    embedder = Embedder()
    pose_est = PoseEst()

    # Gallery embeddings
    log("\nExtracting gallery embeddings...")
    gal_embs = {}
    for sid, path in gallery.items():
        img = cv2.imread(path)
        if img is None:
            continue
        e = embedder.embed(img)
        if e is not None:
            gal_embs[sid] = e
    log(f"  {len(gal_embs)}/{len(gallery)} gallery embeddings OK")

    if not gal_embs:
        log("❌ No gallery embeddings extracted")
        sys.exit(1)

    gal_ids = list(gal_embs.keys())

    # Evaluate
    all_res, all_strat = {}, {}
    for d in ["d1", "d2", "d3"]:
        if not probes[d]:
            log(f"\n  Skip {d}: no probes")
            continue
        log(f"\n{'─' * 60}")
        log(f"  Evaluating {d} ({len(probes[d])} probes)...")

        res = evaluate(gal_embs, gal_ids, probes[d], embedder, pose_est, d)
        all_res[d] = res
        all_strat[d] = stratified(res)

        m = metrics(res)
        log(f"  {d}: Rank-1={m['rank1']:.1%}  Rank-5={m.get('rank5',0):.1%}  Det={m['n_det']}/{m['n']}")
        for bn in ["frontal", "half_profile", "profile"]:
            s = all_strat[d].get(bn, {})
            if s.get("count", 0):
                log(f"    {bn:>14}: Rank-1={s['rank1']:.1%} (n={s['count']})")

    # Output
    log(f"\n{'═' * 70}")
    txt = report(all_res, all_strat)
    print(txt)
    with open(os.path.join(RESULTS, "phase1_report.txt"), "w") as f:
        f.write(txt)

    plots(all_res, all_strat)
    save_json(all_res, all_strat)

    log(f"\n{'═' * 70}")
    log(f"  Phase 1 Complete!")
    log(f"  Report:  {RESULTS}/phase1_report.txt")
    log(f"  Plots:   {RESULTS}/plots/")
    log(f"  JSON:    {RESULTS}/summary.json")
    log(f"{'═' * 70}")


if __name__ == "__main__":
    main()
