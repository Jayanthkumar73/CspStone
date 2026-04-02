#!/usr/bin/env python3
"""
phase0_pose_feasibility.py — Can we estimate head pose from LR surveillance faces?

GO/NO-GO test for PALF-Net.

Fixes from v1:
  - GFPGAN now uses has_aligned=False + pre-upscales tiny images before feeding
  - Real-ESRGAN has explicit min-size guard (skips < 8px)
  - Added "bicubic→GFPGAN" and "RealESRGAN→GFPGAN" combo strategies
  - Verbose logging: confirms when SR actually processes vs falls back
  - Reports per-face SR success/failure counts

Usage:
    python version1/scripts/phase0_pose_feasibility.py --image_dir data/scface
    python version1/scripts/phase0_pose_feasibility.py --synthetic_only
"""

import os, sys, time, argparse, warnings, traceback
warnings.filterwarnings("ignore")

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
THRESHOLD = 15.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════
#  POSE ESTIMATOR
# ═══════════════════════════════════════════════
class PoseEstimator:
    def __init__(self):
        log("Loading 6DRepNet...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._mode = None

        try:
            from sixdrepnet import SixDRepNet
            self.model = SixDRepNet()
            self._mode = "package"
            log(f"  ✅ 6DRepNet ready ({self.device})")
        except Exception as e:
            log(f"  ⚠️  sixdrepnet package: {e}")
            self._try_manual()

    def _try_manual(self):
        w = os.path.join(PRETRAINED, "pose", "6DRepNet_300W_LP_AFLW2000.pth")
        if not os.path.isfile(w):
            log(f"  ❌ No pose weights at {w}")
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
        """Returns (yaw, pitch, roll) in degrees."""
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
                with torch.no_grad():
                    out = self.model(t.to(self.device))
                e = out.cpu().numpy().flatten()
                return float(e[0]), float(e[1]), float(e[2])
        except:
            return None, None, None


# ═══════════════════════════════════════════════
#  SR PROCESSOR — PROPERLY HANDLES ALL SIZES
# ═══════════════════════════════════════════════
class SRProcessor:
    def __init__(self):
        self.gfpgan = None
        self.realesrgan = None
        self.sr_stats = {}  # track success/failure per method

        # ── GFPGAN ──
        gp = os.path.join(PRETRAINED, "sr", "GFPGANv1.4.pth")
        if os.path.isfile(gp):
            try:
                from gfpgan import GFPGANer
                self.gfpgan = GFPGANer(
                    model_path=gp, upscale=4, arch="clean",
                    channel_multiplier=2, bg_upsampler=None)
                log("  ✅ GFPGAN loaded")
            except Exception as e:
                log(f"  ⚠️  GFPGAN: {e}")
        else:
            log(f"  ⚠️  GFPGAN weights not found at {gp}")

        # ── Real-ESRGAN ──
        rp = os.path.join(PRETRAINED, "sr", "RealESRGAN_x4plus.pth")
        if os.path.isfile(rp):
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=4)
                self.realesrgan = RealESRGANer(
                    scale=4, model_path=rp, model=net,
                    tile=0, tile_pad=10, pre_pad=0, half=True)
                log("  ✅ Real-ESRGAN loaded")
            except Exception as e:
                log(f"  ⚠️  Real-ESRGAN: {e}")
        else:
            log(f"  ⚠️  Real-ESRGAN weights not found at {rp}")

        methods = ["bicubic"]
        if self.gfpgan:
            methods.append("gfpgan")
        if self.realesrgan:
            methods.append("realesrgan")
        if self.gfpgan and self.realesrgan:
            methods.append("esrgan_then_gfpgan")
        log(f"  Available methods: {methods}")

    def _track(self, method, success):
        key = method
        if key not in self.sr_stats:
            self.sr_stats[key] = {"ok": 0, "fail": 0}
        self.sr_stats[key]["ok" if success else "fail"] += 1

    def get_methods(self):
        """Return list of available SR method names."""
        methods = ["bicubic"]
        if self.gfpgan:
            methods.append("gfpgan")
        if self.realesrgan:
            methods.append("realesrgan")
        if self.gfpgan and self.realesrgan:
            methods.append("esrgan_then_gfpgan")
        return methods

    def enhance(self, lr_bgr, method, target_size=224):
        """
        Enhance LR face image using specified method.
        Returns (enhanced_image, actually_used_sr: bool)
        """
        h, w = lr_bgr.shape[:2]

        if method == "bicubic":
            out = cv2.resize(lr_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            self._track("bicubic", True)
            return out, False

        elif method == "gfpgan":
            return self._run_gfpgan(lr_bgr, target_size)

        elif method == "realesrgan":
            return self._run_realesrgan(lr_bgr, target_size)

        elif method == "esrgan_then_gfpgan":
            return self._run_esrgan_then_gfpgan(lr_bgr, target_size)

        else:
            out = cv2.resize(lr_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            return out, False

    def _run_gfpgan(self, lr_bgr, target_size):
        """
        GFPGAN needs faces roughly 256-512px.
        Strategy: pre-upscale tiny faces with bicubic, then feed to GFPGAN.
        Use has_aligned=False so it runs its own face detection + restoration.
        """
        if self.gfpgan is None:
            self._track("gfpgan", False)
            return cv2.resize(lr_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC), False

        h, w = lr_bgr.shape[:2]
        try:
            # Pre-upscale to at least 256px so GFPGAN can detect the face
            min_dim = min(h, w)
            if min_dim < 256:
                scale = max(2, 256 // min_dim + 1)
                input_img = cv2.resize(lr_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            else:
                input_img = lr_bgr

            # has_aligned=False: GFPGAN detects face itself → much better results
            _, _, restored = self.gfpgan.enhance(
                input_img, has_aligned=False, only_center_face=True, paste_back=True)

            if restored is not None and restored.size > 0:
                out = cv2.resize(restored, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                self._track("gfpgan", True)
                return out, True

            # If GFPGAN couldn't find a face, try has_aligned=True as fallback
            if min_dim < 512:
                scale2 = max(2, 512 // min_dim + 1)
                input_img2 = cv2.resize(lr_bgr, (w * scale2, h * scale2), interpolation=cv2.INTER_CUBIC)
            else:
                input_img2 = lr_bgr

            _, _, restored2 = self.gfpgan.enhance(
                input_img2, has_aligned=True, only_center_face=True, paste_back=True)

            if restored2 is not None and restored2.size > 0:
                out = cv2.resize(restored2, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                self._track("gfpgan", True)
                return out, True

        except Exception as e:
            pass

        self._track("gfpgan", False)
        return cv2.resize(lr_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC), False

    def _run_realesrgan(self, lr_bgr, target_size):
        """Real-ESRGAN: works on any image, but hallucinates badly below ~10px."""
        if self.realesrgan is None:
            self._track("realesrgan", False)
            return cv2.resize(lr_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC), False

        h, w = lr_bgr.shape[:2]
        try:
            # Compute appropriate outscale
            min_dim = min(h, w)
            if min_dim < 16:
                outscale = max(4, target_size // min_dim)
            else:
                outscale = max(2, target_size // min_dim)
            outscale = min(outscale, 8)  # cap at 8x

            enhanced, _ = self.realesrgan.enhance(lr_bgr, outscale=outscale)
            if enhanced is not None and enhanced.size > 0:
                out = cv2.resize(enhanced, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                self._track("realesrgan", True)
                return out, True

        except Exception as e:
            pass

        self._track("realesrgan", False)
        return cv2.resize(lr_bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC), False

    def _run_esrgan_then_gfpgan(self, lr_bgr, target_size):
        """
        Two-stage: Real-ESRGAN first (generic upscale), then GFPGAN (face-specific).
        This combo often gives best results on very small faces.
        """
        # Stage 1: Real-ESRGAN to get a reasonable-sized image
        h, w = lr_bgr.shape[:2]
        stage1 = lr_bgr

        if self.realesrgan is not None:
            try:
                outscale = max(4, 256 // min(h, w) + 1)
                outscale = min(outscale, 8)
                stage1, _ = self.realesrgan.enhance(lr_bgr, outscale=outscale)
            except:
                stage1 = cv2.resize(lr_bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        else:
            stage1 = cv2.resize(lr_bgr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

        # Stage 2: GFPGAN on the upscaled result
        if self.gfpgan is not None:
            try:
                _, _, restored = self.gfpgan.enhance(
                    stage1, has_aligned=False, only_center_face=True, paste_back=True)
                if restored is not None and restored.size > 0:
                    out = cv2.resize(restored, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                    self._track("esrgan_then_gfpgan", True)
                    return out, True

                # Fallback: has_aligned=True
                _, _, restored2 = self.gfpgan.enhance(
                    stage1, has_aligned=True, only_center_face=True, paste_back=True)
                if restored2 is not None and restored2.size > 0:
                    out = cv2.resize(restored2, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                    self._track("esrgan_then_gfpgan", True)
                    return out, True
            except:
                pass

        # Fallback to just the ESRGAN result
        out = cv2.resize(stage1, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        self._track("esrgan_then_gfpgan", False)
        return out, False


# ═══════════════════════════════════════════════
#  TEST FACE SOURCES
# ═══════════════════════════════════════════════
def make_synthetic_faces(n=20):
    log(f"Generating {n} synthetic test faces...")
    faces = []
    for i in range(n):
        img = np.full((112, 112, 3), 200, dtype=np.uint8)
        cx, cy = 56 + np.random.randint(-8, 8), 56
        cv2.ellipse(img, (cx, cy), (35, 45), 0, 0, 360, (180, 150, 130), -1)
        shift = np.random.randint(-10, 10)
        cv2.circle(img, (40 + shift, 44), 5, (40, 40, 40), -1)
        cv2.circle(img, (72 + shift, 44), 5, (40, 40, 40), -1)
        cv2.circle(img, (56 + shift, 60), 3, (140, 120, 100), -1)
        cv2.ellipse(img, (56 + shift, 76), (12, 4), 0, 0, 360, (120, 80, 80), -1)
        faces.append(img)
    return faces


def load_real_faces(d, n=50):
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
def test_one(pose_est, sr_proc, hr_face, res, methods):
    """Test pose estimation at one resolution with all SR methods."""
    # Ground truth: pose from HR (112x112) upscaled to 224
    gt_y, gt_p, gt_r = pose_est.predict(cv2.resize(hr_face, (224, 224)))
    if gt_y is None:
        return None

    # Downsample to test resolution
    lr = cv2.resize(hr_face, (res, res), interpolation=cv2.INTER_AREA)

    out = {"gt": (gt_y, gt_p, gt_r)}

    for method in methods:
        enhanced, sr_used = sr_proc.enhance(lr, method=method, target_size=224)
        y, p, r = pose_est.predict(enhanced)
        if y is not None:
            out[method] = {
                "yaw": y, "pitch": p, "roll": r,
                "yaw_err": abs(y - gt_y),
                "pitch_err": abs(p - gt_p),
                "roll_err": abs(r - gt_r),
                "sr_used": sr_used,
            }

    return out


def run_all_tests(pose_est, sr_proc, faces):
    methods = sr_proc.get_methods()
    log(f"\nTesting {len(faces)} faces × {len(TEST_RES)} resolutions × {len(methods)} methods")
    log(f"Methods: {methods}")
    log("-" * 80)

    data = {}
    for res in TEST_RES:
        data[res] = {m: [] for m in methods}

        for face in faces:
            r = test_one(pose_est, sr_proc, face, res, methods)
            if r is None:
                continue
            for m in methods:
                if m in r:
                    data[res][m].append(r[m])

        # Progress log
        n = len(data[res].get("bicubic", []))
        if n:
            parts = []
            for m in methods:
                if data[res][m]:
                    err = np.mean([x["yaw_err"] for x in data[res][m]])
                    sr_count = sum(1 for x in data[res][m] if x.get("sr_used"))
                    parts.append(f"{m}={err:.1f}°({sr_count}/{len(data[res][m])} SR)")
                else:
                    parts.append(f"{m}=N/A")

            best_err = 999
            for m in methods:
                if data[res][m]:
                    e = np.mean([x["yaw_err"] for x in data[res][m]])
                    best_err = min(best_err, e)

            ok = "✅" if best_err < THRESHOLD else "❌"
            log(f"  {res:3d}×{res:3d}  n={n:3d}  {' | '.join(parts)}  {ok}")

    # Print SR stats
    log(f"\n  SR processing stats:")
    for method, stats in sr_proc.sr_stats.items():
        total = stats["ok"] + stats["fail"]
        log(f"    {method:>20s}: {stats['ok']}/{total} succeeded "
            f"({100*stats['ok']/total:.0f}%)" if total > 0 else f"    {method}: 0 calls")

    return data


# ═══════════════════════════════════════════════
#  REPORT & PLOTS
# ═══════════════════════════════════════════════
def make_report(data, faces, sr_proc):
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "sr_samples"), exist_ok=True)

    methods = sr_proc.get_methods()

    lines = [
        "=" * 80,
        "  PALF-Net Phase 0 — Pose Estimation Feasibility Report (v2)",
        "=" * 80,
        f"  Faces tested: {len(faces)}",
        f"  Threshold: yaw error < {THRESHOLD}°",
        f"  SR methods: {methods}",
        "",
    ]

    # Header
    hdr = f"{'Res':>8}"
    for m in methods:
        hdr += f" | {m:>16s}"
    hdr += f" | {'Best':>16s} | {'Feasible':>8s}"
    lines.append(hdr)
    lines.append("-" * (len(hdr) + 5))

    feasibility = {}
    for res in TEST_RES:
        best_e, best_m = 999, "N/A"
        parts = []
        for m in methods:
            if data[res][m]:
                e = np.mean([x["yaw_err"] for x in data[res][m]])
                sr_ct = sum(1 for x in data[res][m] if x.get("sr_used"))
                parts.append(f"{e:>8.1f}° [{sr_ct}sr]")
                if e < best_e:
                    best_e, best_m = e, m
            else:
                parts.append(f"{'N/A':>14s}")

        ok = best_e < THRESHOLD
        feasibility[res] = (ok, best_e, best_m)
        line = f"{res:>4}×{res:<3}"
        for p in parts:
            line += f" | {p}"
        line += f" | {best_e:>8.1f}° {best_m:>5s} | {'YES ✅' if ok else 'NO  ❌':>8s}"
        lines.append(line)

    # Decision
    crit = [16, 24, 32]
    crit_ok = sum(1 for r in crit if r in feasibility and feasibility[r][0])

    lines += ["", "=" * 80, "  DECISION", "=" * 80]

    if crit_ok == len(crit):
        decision = "GO ✅"
        lines.append(f"  {decision} — Pose estimation feasible at all critical resolutions!")
        lines.append("  Recommended pipeline: LR → Real-ESRGAN → 6DRepNet → FiLM conditioning")
    elif crit_ok > 0:
        decision = "CONDITIONAL GO ✅"
        lines.append(f"  {decision} — Feasible at {crit_ok}/{len(crit)} critical resolutions")
        for r in crit:
            if r in feasibility:
                ok, e, m = feasibility[r]
                lines.append(f"    {r}×{r}: best={m} err={e:.1f}° {'✅' if ok else '❌'}")
    else:
        decision = "NO-GO ❌"
        lines.append(f"  {decision} — Pose estimation fails at surveillance resolutions")
        lines.append("  Consider: gallery-proxy pose, bbox aspect ratio, or trained LR pose classifier")

    # Best method per resolution
    lines += ["", "  Best SR method per resolution:"]
    for res in TEST_RES:
        if res in feasibility:
            ok, e, m = feasibility[res]
            lines.append(f"    {res:>3}×{res:<3}: {m:>20s}  err={e:.1f}° {'✅' if ok else '❌'}")

    # SR impact analysis
    lines += ["", "  SR Impact (improvement over bicubic):"]
    for res in [8, 12, 16, 24, 32]:
        if data[res].get("bicubic"):
            be = np.mean([x["yaw_err"] for x in data[res]["bicubic"]])
            for m in methods:
                if m == "bicubic":
                    continue
                if data[res].get(m):
                    me = np.mean([x["yaw_err"] for x in data[res][m]])
                    diff = be - me
                    sr_ct = sum(1 for x in data[res][m] if x.get("sr_used"))
                    tag = "HELPS" if diff > 2 else ("HURTS" if diff < -2 else "~same")
                    lines.append(f"    {res:>3}×{res:<3} {m:>20s}: {diff:>+6.1f}° ({tag}) "
                                 f"[SR ran on {sr_ct}/{len(data[res][m])}]")

    lines.append("\n" + "=" * 80)
    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(os.path.join(RESULTS, "phase0_report.txt"), "w") as f:
        f.write(report_text)

    # ─── Plots ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    color_map = {
        "bicubic": "red", "gfpgan": "blue",
        "realesrgan": "green", "esrgan_then_gfpgan": "purple",
    }

    for m in methods:
        xs, ys = [], []
        for res in TEST_RES:
            if data[res][m]:
                xs.append(res)
                ys.append(np.mean([x["yaw_err"] for x in data[res][m]]))
        if xs:
            c = color_map.get(m, "gray")
            ax1.plot(xs, ys, "o-", color=c, label=m, lw=2, ms=8)

    ax1.axhline(THRESHOLD, color="gray", ls="--", alpha=.6, label=f"Threshold ({THRESHOLD}°)")
    ax1.set_xlabel("Resolution (px)", fontsize=11)
    ax1.set_ylabel("Mean Yaw Error (°)", fontsize=11)
    ax1.set_title("Pose Error vs Resolution (All SR Methods)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=.3)
    ax1.invert_xaxis()
    ax1.set_xticks(TEST_RES)

    for angle, mk, c in [("yaw", "o", "#e74c3c"), ("pitch", "s", "#3498db"), ("roll", "^", "#2ecc71")]:
        # Use best method at each resolution
        xs, ys = [], []
        for res in TEST_RES:
            best_err, best_method = 999, "bicubic"
            for m in methods:
                if data[res][m]:
                    e = np.mean([x["yaw_err"] for x in data[res][m]])
                    if e < best_err:
                        best_err, best_method = e, m
            if data[res].get(best_method):
                xs.append(res)
                ys.append(np.mean([x[f"{angle}_err"] for x in data[res][best_method]]))
        if xs:
            ax2.plot(xs, ys, f"{mk}-", color=c, label=angle, lw=2, ms=8)

    ax2.axhline(THRESHOLD, color="gray", ls="--", alpha=.6)
    ax2.set_xlabel("Resolution (px)", fontsize=11)
    ax2.set_ylabel("Mean Error (°)", fontsize=11)
    ax2.set_title("Yaw / Pitch / Roll (Best Method per Res)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=.3)
    ax2.invert_xaxis()
    ax2.set_xticks(TEST_RES)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "visualizations", "pose_vs_resolution.png"), dpi=150)
    plt.close()
    log(f"Plot saved: {RESULTS}/visualizations/pose_vs_resolution.png")

    # SR visual comparison samples
    save_sr_samples(faces[:5], sr_proc, methods)

    return decision


def save_sr_samples(faces, sr_proc, methods):
    if not faces:
        return
    resolutions = [8, 12, 16, 24, 32]

    for idx, hr in enumerate(faces):
        n_cols = 1 + len(methods)  # LR + each method
        fig, axes = plt.subplots(len(resolutions), n_cols,
                                 figsize=(3 * n_cols, 3 * len(resolutions)))
        if len(resolutions) == 1:
            axes = axes.reshape(1, -1)

        for ri, res in enumerate(resolutions):
            lr = cv2.resize(hr, (res, res), interpolation=cv2.INTER_AREA)
            vis = cv2.resize(lr, (112, 112), interpolation=cv2.INTER_NEAREST)
            axes[ri, 0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            axes[ri, 0].set_title(f"LR {res}×{res}" if ri == 0 else f"{res}×{res}")
            axes[ri, 0].axis("off")

            for mi, m in enumerate(methods):
                enhanced, sr_used = sr_proc.enhance(lr, method=m, target_size=112)
                axes[ri, mi + 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                tag = "✓SR" if sr_used else "✗fall"
                if ri == 0:
                    axes[ri, mi + 1].set_title(f"{m}\n({tag})")
                else:
                    axes[ri, mi + 1].set_title(f"({tag})", fontsize=8)
                axes[ri, mi + 1].axis("off")

        plt.suptitle(f"SR Comparison — Sample {idx + 1}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "sr_samples", f"sample_{idx+1}.png"), dpi=120)
        plt.close()

    log(f"SR samples saved: {RESULTS}/sr_samples/")


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Phase 0: Pose Estimation Feasibility (v2)")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--synthetic_only", action="store_true")
    parser.add_argument("--max_faces", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    log("=" * 60)
    log("  PALF-Net Phase 0 — Pose Feasibility Test (v2)")
    log("=" * 60)

    pose_est = PoseEstimator()
    if pose_est.model is None:
        log("FATAL: No pose estimator. Run: python scripts/download_models.py --model pose")
        sys.exit(1)

    sr_proc = SRProcessor()

    faces = []
    if args.image_dir:
        faces = load_real_faces(args.image_dir, args.max_faces)
    if not faces and not args.synthetic_only:
        found = find_any_images()
        if found:
            log(f"Found images in: {found}")
            faces = load_real_faces(found, args.max_faces)
    if not faces:
        log("No real images → synthetic (re-run with --image_dir for real results)")
        faces = make_synthetic_faces(args.max_faces)

    data = run_all_tests(pose_est, sr_proc, faces)
    decision = make_report(data, faces, sr_proc)

    log(f"\n{'=' * 60}")
    log(f"  DONE — Decision: {decision}")
    log(f"  Report: {RESULTS}/phase0_report.txt")
    log(f"  Plot:   {RESULTS}/visualizations/pose_vs_resolution.png")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
