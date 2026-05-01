#!/usr/bin/env python3
"""
phase0_pose_feasibility.py — Can we estimate head pose from 15-50px surveillance faces?

This is the GO/NO-GO test for the entire PALF-Net project.

Usage:
    python version1/scripts/phase0_pose_feasibility.py

    # Point to real face images (SCface, any face folder):
    python version1/scripts/phase0_pose_feasibility.py --image_dir version1/data/scface

    # Synthetic only (no dataset needed):
    python version1/scripts/phase0_pose_feasibility.py --synthetic_only
"""

import os, sys, time, argparse, warnings
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "phase0")
PRETRAINED = os.path.join(ROOT, "pretrained")

TEST_RES = [112, 64, 48, 32, 24, 16, 12, 8]
THRESHOLD = 15.0  # degrees — feasibility cutoff


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════
#  MODEL WRAPPERS
# ═══════════════════════════════════════════════
class PoseEstimator:
    def __init__(self):
        log("Loading 6DRepNet...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        try:
            from sixdrepnet import SixDRepNet
            self.model = SixDRepNet()
            self._mode = "package"
            log(f"  ✅ 6DRepNet ready ({self.device})")
        except Exception as e:
            log(f"  ❌ sixdrepnet package failed: {e}")
            self._try_manual_load()

    def _try_manual_load(self):
        w = os.path.join(PRETRAINED, "pose", "6DRepNet_300W_LP_AFLW2000.pth")
        if not os.path.isfile(w):
            log(f"  ❌ Weight file not found: {w}")
            return
        try:
            from sixdrepnet.model import SixDRepNet as M
            self.model = M(backbone_name="RepVGG-B1g2", backbone_file="", deploy=True, pretrained=False)
            sd = torch.load(w, map_location=self.device)
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            self.model.load_state_dict(sd, strict=False)
            self.model.to(self.device).eval()
            self._mode = "manual"
            log("  ✅ 6DRepNet loaded (manual)")
        except Exception as e:
            log(f"  ❌ Manual load failed: {e}")

    def predict(self, bgr_img):
        """Returns (yaw, pitch, roll) in degrees or (None,None,None)."""
        if self.model is None:
            return None, None, None
        try:
            if self._mode == "package":
                y, p, r = self.model.predict(bgr_img)
                if isinstance(y, (list, np.ndarray)):
                    y, p, r = float(y[0]), float(p[0]), float(r[0])
                return float(y), float(p), float(r)
            else:
                rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                t = t.to(self.device)
                with torch.no_grad():
                    out = self.model(t)
                e = out.cpu().numpy().flatten()
                return float(e[0]), float(e[1]), float(e[2])
        except:
            return None, None, None


class SRProcessor:
    def __init__(self):
        self.methods = {}
        # GFPGAN
        gp = os.path.join(PRETRAINED, "sr", "GFPGANv1.4.pth")
        if os.path.isfile(gp):
            try:
                from gfpgan import GFPGANer
                self.methods["gfpgan"] = GFPGANer(
                    model_path=gp, upscale=1, arch="clean",
                    channel_multiplier=2, bg_upsampler=None)
                log("  ✅ GFPGAN loaded")
            except Exception as e:
                log(f"  ⚠️  GFPGAN: {e}")

        # Real-ESRGAN
        rp = os.path.join(PRETRAINED, "sr", "RealESRGAN_x4plus.pth")
        if os.path.isfile(rp):
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)
                self.methods["realesrgan"] = RealESRGANer(
                    scale=4, model_path=rp, model=net,
                    tile=0, tile_pad=10, pre_pad=0, half=True)
                log("  ✅ Real-ESRGAN loaded")
            except Exception as e:
                log(f"  ⚠️  Real-ESRGAN: {e}")

        if not self.methods:
            log("  ⚠️  No SR models loaded — bicubic only")

    def enhance(self, lr_bgr, method="gfpgan", size=224):
        if method == "bicubic" or method not in self.methods:
            return cv2.resize(lr_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
        try:
            if method == "gfpgan":
                _, _, out = self.methods["gfpgan"].enhance(lr_bgr, has_aligned=True, only_center_face=True)
            elif method == "realesrgan":
                out, _ = self.methods["realesrgan"].enhance(lr_bgr, outscale=4)
            return cv2.resize(out, (size, size), interpolation=cv2.INTER_LINEAR)
        except:
            return cv2.resize(lr_bgr, (size, size), interpolation=cv2.INTER_CUBIC)


# ═══════════════════════════════════════════════
#  TEST FACE SOURCES
# ═══════════════════════════════════════════════
def make_synthetic_faces(n=20):
    """Procedural face-like images for testing when no dataset is available."""
    log(f"Generating {n} synthetic test faces...")
    faces = []
    for i in range(n):
        img = np.full((112, 112, 3), 200, dtype=np.uint8)
        # face oval
        cx, cy = 56 + np.random.randint(-8, 8), 56
        cv2.ellipse(img, (cx, cy), (35, 45), 0, 0, 360, (180, 150, 130), -1)
        # eyes
        shift = np.random.randint(-10, 10)  # simulates pose
        cv2.circle(img, (40 + shift, 44), 5, (40, 40, 40), -1)
        cv2.circle(img, (72 + shift, 44), 5, (40, 40, 40), -1)
        # nose, mouth
        cv2.circle(img, (56 + shift, 60), 3, (140, 120, 100), -1)
        cv2.ellipse(img, (56 + shift, 76), (12, 4), 0, 0, 360, (120, 80, 80), -1)
        faces.append(img)
    return faces


def load_real_faces(d, n=50):
    """Load face images from any directory."""
    log(f"Scanning {d} for face images...")
    faces = []
    for dp, _, fns in os.walk(d):
        for f in sorted(fns):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = cv2.imread(os.path.join(dp, f))
            if img is None:
                continue
            h, w = img.shape[:2]
            s = min(h, w)
            img = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
            faces.append(cv2.resize(img, (112, 112)))
            if len(faces) >= n:
                return faces
    log(f"  Found {len(faces)} images")
    return faces


def find_any_images():
    """Auto-search for face images in data/."""
    for sub in ["scface", "tinyface", "survface", "training"]:
        d = os.path.join(ROOT, "data", sub)
        if not os.path.isdir(d):
            continue
        for dp, _, fns in os.walk(d):
            if any(f.lower().endswith(('.jpg', '.png')) for f in fns):
                return dp
    return None


# ═══════════════════════════════════════════════
#  CORE TEST
# ═══════════════════════════════════════════════
def test_one(pose_est, sr_proc, hr_face, res):
    """Test pose estimation at one resolution with multiple SR methods."""
    gt_y, gt_p, gt_r = pose_est.predict(cv2.resize(hr_face, (224, 224)))
    if gt_y is None:
        return None

    lr = cv2.resize(hr_face, (res, res), interpolation=cv2.INTER_AREA)
    out = {"gt": (gt_y, gt_p, gt_r)}

    for method in ["bicubic", "gfpgan", "realesrgan"]:
        up = sr_proc.enhance(lr, method=method, size=224)
        y, p, r = pose_est.predict(up)
        if y is not None:
            out[method] = {
                "yaw": y, "pitch": p, "roll": r,
                "yaw_err": abs(y - gt_y),
                "pitch_err": abs(p - gt_p),
                "roll_err": abs(r - gt_r),
            }
    return out


def run_all_tests(pose_est, sr_proc, faces):
    log(f"\nTesting {len(faces)} faces × {len(TEST_RES)} resolutions × {1+len(sr_proc.methods)} methods")
    log("-" * 70)

    data = {}
    for res in TEST_RES:
        data[res] = {"bicubic": [], "gfpgan": [], "realesrgan": []}
        for face in faces:
            r = test_one(pose_est, sr_proc, face, res)
            if r is None:
                continue
            for m in ["bicubic", "gfpgan", "realesrgan"]:
                if m in r:
                    data[res][m].append(r[m])

        # progress
        n = len(data[res]["bicubic"])
        if n:
            be = np.mean([x["yaw_err"] for x in data[res]["bicubic"]])
            parts = [f"bicubic={be:.1f}°"]
            for m in ["gfpgan", "realesrgan"]:
                if data[res][m]:
                    parts.append(f"{m}={np.mean([x['yaw_err'] for x in data[res][m]]):.1f}°")
            ok = "✅" if be < THRESHOLD else "❌"
            log(f"  {res:3d}×{res:3d}  n={n:3d}  {' | '.join(parts)}  {ok}")

    return data


# ═══════════════════════════════════════════════
#  REPORT & PLOTS
# ═══════════════════════════════════════════════
def make_report(data, faces, sr_proc):
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "sr_samples"), exist_ok=True)

    lines = [
        "=" * 70,
        "  PALF-Net Phase 0 — Pose Estimation Feasibility Report",
        "=" * 70,
        f"  Faces tested: {len(faces)}",
        f"  Threshold: yaw error < {THRESHOLD}°",
        f"  SR methods: {['bicubic'] + list(sr_proc.methods.keys())}",
        "",
        f"{'Res':>8} | {'Bicubic':>10} | {'GFPGAN':>10} | {'RealESRGAN':>12} | {'Best':>10} | {'Feasible':>8}",
        "-" * 70,
    ]

    feasibility = {}
    for res in TEST_RES:
        best_e, best_m = 999, "N/A"
        parts = []
        for m in ["bicubic", "gfpgan", "realesrgan"]:
            if data[res][m]:
                e = np.mean([x["yaw_err"] for x in data[res][m]])
                parts.append(f"{e:>8.1f}°")
                if e < best_e:
                    best_e, best_m = e, m
            else:
                parts.append(f"{'N/A':>8s} ")
        ok = best_e < THRESHOLD
        feasibility[res] = (ok, best_e, best_m)
        lines.append(f"{res:>4}×{res:<3} | {' | '.join(parts)} | {best_e:>8.1f}° | {'YES ✅' if ok else 'NO  ❌':>8s}")

    # Decision
    crit = [16, 24, 32]
    crit_ok = sum(1 for r in crit if r in feasibility and feasibility[r][0])

    lines += ["", "=" * 70, "  DECISION", "=" * 70]
    if crit_ok == len(crit):
        decision = "GO ✅"
        lines.append(f"  {decision} — Pose estimation feasible at all surveillance resolutions")
        lines.append(f"  → Use direct pose estimation on LR faces for FiLM conditioning")
    elif crit_ok > 0:
        decision = "CONDITIONAL GO ✅"
        lines.append(f"  {decision} — Works at some resolutions, use SR preprocessing for smallest")
        for r in crit:
            if r in feasibility:
                ok, e, m = feasibility[r]
                lines.append(f"    {r}×{r}: best={m} err={e:.1f}° {'✅' if ok else '❌'}")
    else:
        decision = "NO-GO ❌"
        lines.append(f"  {decision} — Pose estimation fails at surveillance resolutions")
        lines.append("  Alternatives:")
        lines.append("    1. Use HR gallery image pose as proxy")
        lines.append("    2. Use bbox aspect ratio as rough pose signal")
        lines.append("    3. Train pose classifier on SR'd images")

    # SR recommendation
    lines += ["", "  SR Impact:"]
    for res in [16, 24, 32]:
        if data[res]["bicubic"] and data[res].get("gfpgan"):
            be = np.mean([x["yaw_err"] for x in data[res]["bicubic"]])
            ge = np.mean([x["yaw_err"] for x in data[res]["gfpgan"]])
            diff = be - ge
            if diff > 1:
                lines.append(f"    {res}×{res}: GFPGAN reduces error by {diff:.1f}° → USE SR FIRST")
            else:
                lines.append(f"    {res}×{res}: GFPGAN gives {diff:+.1f}° → bicubic sufficient")

    lines.append("\n" + "=" * 70)
    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(RESULTS, "phase0_report.txt"), "w") as f:
        f.write(report)

    # ─── Plot ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"bicubic": "red", "gfpgan": "blue", "realesrgan": "green"}
    for m, c in colors.items():
        xs, ys = [], []
        for res in TEST_RES:
            if data[res][m]:
                xs.append(res)
                ys.append(np.mean([x["yaw_err"] for x in data[res][m]]))
        if xs:
            ax1.plot(xs, ys, "o-", color=c, label=m, lw=2, ms=8)

    ax1.axhline(THRESHOLD, color="gray", ls="--", alpha=.6, label=f"Threshold ({THRESHOLD}°)")
    ax1.set_xlabel("Resolution (px)")
    ax1.set_ylabel("Mean Yaw Error (°)")
    ax1.set_title("Pose Estimation Error vs Resolution")
    ax1.legend()
    ax1.grid(alpha=.3)
    ax1.invert_xaxis()
    ax1.set_xticks(TEST_RES)

    for angle, mk in [("yaw", "o"), ("pitch", "s"), ("roll", "^")]:
        xs, ys = [], []
        for res in TEST_RES:
            if data[res]["bicubic"]:
                xs.append(res)
                ys.append(np.mean([x[f"{angle}_err"] for x in data[res]["bicubic"]]))
        if xs:
            ax2.plot(xs, ys, f"{mk}-", label=angle, lw=2, ms=8)

    ax2.axhline(THRESHOLD, color="gray", ls="--", alpha=.6)
    ax2.set_xlabel("Resolution (px)")
    ax2.set_ylabel("Mean Error (°)")
    ax2.set_title("Yaw / Pitch / Roll (Bicubic)")
    ax2.legend()
    ax2.grid(alpha=.3)
    ax2.invert_xaxis()
    ax2.set_xticks(TEST_RES)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "visualizations", "pose_vs_resolution.png"), dpi=150)
    plt.close()
    log(f"Plot saved: {RESULTS}/visualizations/pose_vs_resolution.png")

    # ─── SR visual samples ───
    save_sr_samples(faces[:5], sr_proc)

    return decision


def save_sr_samples(faces, sr_proc):
    if not faces:
        return
    methods = ["bicubic"] + list(sr_proc.methods.keys())
    resolutions = [16, 24, 32]

    for idx, hr in enumerate(faces):
        fig, axes = plt.subplots(len(resolutions), len(methods) + 1,
                                 figsize=(3 * (len(methods) + 1), 3 * len(resolutions)))
        if len(resolutions) == 1:
            axes = axes.reshape(1, -1)

        for ri, res in enumerate(resolutions):
            lr = cv2.resize(hr, (res, res), interpolation=cv2.INTER_AREA)
            vis = cv2.resize(lr, (112, 112), interpolation=cv2.INTER_NEAREST)
            axes[ri, 0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[ri, 0].set_title(f"LR {res}×{res}" if ri == 0 else f"{res}×{res}")
            axes[ri, 0].axis("off")

            for mi, m in enumerate(methods):
                sr = sr_proc.enhance(lr, method=m, size=112)
                axes[ri, mi + 1].imshow(cv2.cvtColor(sr, cv2.COLOR_BGR2RGB))
                if ri == 0:
                    axes[ri, mi + 1].set_title(m.upper())
                axes[ri, mi + 1].axis("off")

        plt.suptitle(f"SR Comparison — Sample {idx + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "sr_samples", f"sample_{idx+1}.png"), dpi=120)
        plt.close()

    log(f"SR samples saved: {RESULTS}/sr_samples/")


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Phase 0: Pose Estimation Feasibility")
    parser.add_argument("--image_dir", type=str, default=None, help="Folder with face images")
    parser.add_argument("--synthetic_only", action="store_true", help="Skip real image search")
    parser.add_argument("--max_faces", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    log("=" * 60)
    log("  PALF-Net Phase 0 — Pose Feasibility Test")
    log("=" * 60)

    # Models
    pose_est = PoseEstimator()
    if pose_est.model is None:
        log("FATAL: No pose estimator. Run: python scripts/download_models.py --model pose")
        sys.exit(1)

    sr_proc = SRProcessor()

    # Faces
    faces = []
    if args.image_dir:
        faces = load_real_faces(args.image_dir, args.max_faces)
    if not faces and not args.synthetic_only:
        found = find_any_images()
        if found:
            log(f"Found images in: {found}")
            faces = load_real_faces(found, args.max_faces)
    if not faces:
        log("No real images found → using synthetic (re-run with --image_dir for accurate results)")
        faces = make_synthetic_faces(args.max_faces)

    # Run
    data = run_all_tests(pose_est, sr_proc, faces)
    decision = make_report(data, faces, sr_proc)

    log(f"\n{'=' * 60}")
    log(f"  DONE — Decision: {decision}")
    log(f"  Report: {RESULTS}/phase0_report.txt")
    log(f"  Plot:   {RESULTS}/visualizations/pose_vs_resolution.png")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
