"""
generate_gate_heatmap.py  (v3 — robust path detection)
-------------------------------------------------------
Run from project root:
    python scripts/generate_gate_heatmap.py \
        --checkpoint experiments/adaface_sr_v2_clean/checkpoints/checkpoint_best.pth \
        --scface_root data/scface \
        --output_dir  results/figures
"""

import argparse, sys, os
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.adaface_sr import AdaFaceSR


# ── path finder ───────────────────────────────────────────────────────────────

def find_scface_d3_dir(root_str):
    """
    Returns the d3 probe directory.
    Confirmed layout: data/scface/organized/surveillance/d3/
    """
    root = Path(root_str)
    candidates = [
        root / "organized" / "surveillance" / "d3",   # confirmed
        root / "organized" / "surveillance" / "d3" ,
        root / "surveillance" / "d3",
        root / "surveillance_cameras_distance_3",
        root / "SCface_database" / "surveillance_cameras_distance_3",
    ]
    # Broad fallback
    for p in root.rglob("*"):
        if p.is_dir() and p.name == "d3":
            candidates.append(p)

    print(f"\nSearching for SCface d3 directory under: {root}")
    for c in candidates:
        jpgs = list(c.glob("*.jpg")) if c.exists() else []
        print(f"  {c}: {'EXISTS' if c.exists() else 'not found'}"
              + (f", {len(jpgs)} jpgs" if c.exists() else ""))
        if jpgs:
            print(f"    -> Using this directory.")
            return c

    # Last resort: find ANY jpg under root and use its parent
    all_jpgs = list(root.rglob("*.jpg"))
    if all_jpgs:
        fallback = all_jpgs[0].parent
        print(f"\n  Fallback: using first jpg directory found: {fallback}")
        return fallback

    raise FileNotFoundError(
        f"Cannot find SCface surveillance images under {root}.\n"
        f"Run:  ls {root}/  and pass the correct path via --scface_root.\n"
        f"Alternatively use --probe_path to point directly to a single jpg."
    )


def find_probe(scface_root, probe_path=None):
    """Return a BGR probe image."""
    if probe_path:
        img = cv2.imread(probe_path)
        if img is not None:
            print(f"Using probe: {probe_path}")
            return img
        raise FileNotFoundError(f"Cannot read --probe_path: {probe_path}")

    d3_dir = find_scface_d3_dir(scface_root)
    for p in sorted(d3_dir.glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is not None and img.shape[0] > 10:
            print(f"Using probe: {p}")
            return img
    raise FileNotFoundError(f"No readable jpg in {d3_dir}")


# ── image utilities ───────────────────────────────────────────────────────────

def bgr_to_rgb_u8(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def tensor_to_rgb_u8(t):
    img = t.squeeze(0).permute(1,2,0).cpu().float().numpy()
    return ((img * 0.5 + 0.5).clip(0,1) * 255).astype(np.uint8)

def load_as_tensor(img_bgr, size, device):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size),
                     interpolation=cv2.INTER_AREA if size < 64 else cv2.INTER_CUBIC)
    t = torch.from_numpy(rgb).float().permute(2,0,1) / 255.0
    return ((t - 0.5) / 0.5).unsqueeze(0).to(device)


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(rows_data, output_path, alpha_vmax=None):
    """
    rows_data: list of dicts — label, bicubic_rgb, alpha_map,
               alpha_mean, delta_map, output_rgb
    alpha_vmax: if None, auto-set to max(alpha) across all rows × 1.1
    """
    if alpha_vmax is None:
        all_max = max(float(d["alpha_map"].max()) for d in rows_data)
        alpha_vmax = max(all_max * 1.1, 1e-4)   # at least a tiny range

    n = len(rows_data)
    col_titles = ["Bicubic input",
                  r"Gate $\bar{\alpha}$  (cool–warm)",
                  r"SR correction $|\Delta|$  (hot)",
                  r"Output $\hat{I}$"]

    fig = plt.figure(figsize=(11.0, 3.1 * n + 0.8))
    gs  = gridspec.GridSpec(n, 4, wspace=0.04, hspace=0.16,
                            left=0.10, right=0.91, top=0.94, bottom=0.03)

    for col, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
        ax.axis("off")

    im_ref = None
    for row, d in enumerate(rows_data):
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(d["bicubic_rgb"]); ax0.axis("off")
        ax0.set_ylabel(d["label"], fontsize=8.5, rotation=90,
                       labelpad=3, va="center")

        ax1 = fig.add_subplot(gs[row, 1])
        im = ax1.imshow(d["alpha_map"], cmap="coolwarm",
                        vmin=0, vmax=alpha_vmax, interpolation="nearest")
        ax1.axis("off")
        ax1.text(0.97, 0.03,
                 r"$\bar{\alpha}$=" + f"{d['alpha_mean']:.5f}",
                 transform=ax1.transAxes, ha="right", va="bottom",
                 fontsize=7.5, color="white",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        im_ref = im

        ax2 = fig.add_subplot(gs[row, 2])
        dv = max(float(d["delta_map"].max()), 1e-6)
        ax2.imshow(d["delta_map"], cmap="hot", vmin=0, vmax=dv,
                   interpolation="nearest")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[row, 3])
        ax3.imshow(d["output_rgb"]); ax3.axis("off")

    if im_ref is not None:
        cbar_ax = fig.add_axes([0.924, 0.12, 0.014, 0.74])
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        cbar.set_label(r"$\alpha$ value", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        # Add note if scale is very small
        if alpha_vmax < 0.02:
            cbar.ax.text(0.5, -0.06,
                         f"max={alpha_vmax:.4f}", transform=cbar.ax.transAxes,
                         ha="center", fontsize=6.5, color="#555555")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(str(output_path) + ext, dpi=250, bbox_inches="tight")
        print(f"Saved: {output_path}{ext}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=
        "experiments/adaface_sr_v2_clean/checkpoints/checkpoint_best.pth")
    parser.add_argument("--scface_root",  default="data/scface")
    parser.add_argument("--probe_path",   default=None,
        help="Direct path to a single jpg probe (bypasses scface_root search).")
    parser.add_argument("--output_dir",   default="results/figures")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading: {args.checkpoint}")
    model = AdaFaceSR().to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    st = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(st, strict=False)
    model.eval()

    # Quick alpha diagnostic
    with torch.no_grad():
        dummy = torch.zeros(1,3,112,112, device=device)
        _, alpha_d = model(dummy, torch.tensor([16.0], device=device))
    print(f"Alpha diagnostic: mean={float(alpha_d.mean()):.6f}  "
          f"max={float(alpha_d.max()):.6f}")

    # Load probe
    probe_bgr = find_probe(args.scface_root, args.probe_path)

    # Run at three resolutions
    rows_data = []
    configs = [(16, "16\u202fpx input"),
               (24, "24\u202fpx input"),
               (112, "Native input")]

    with torch.no_grad():
        for r_val, label in configs:
            small = (cv2.resize(probe_bgr, (r_val, r_val),
                                interpolation=cv2.INTER_AREA)
                     if r_val < 112 else probe_bgr)
            bic_bgr = cv2.resize(small, (112,112), interpolation=cv2.INTER_CUBIC)
            bic_t   = load_as_tensor(bic_bgr, 112, device)
            lr_sz   = torch.tensor([float(r_val)], device=device)

            out_t, alpha_t = model(bic_t, lr_sz)

            # Delta via direct branch call
            res_emb = model.res_encoder(lr_sz)
            delta_t = model.sr_branch(bic_t, res_emb)

            alpha_np = alpha_t.squeeze(0).mean(0).cpu().numpy()
            delta_np = delta_t.squeeze(0).abs().mean(0).cpu().numpy()
            a_mean   = float(alpha_np.mean())

            print(f"  {label}: alpha_mean={a_mean:.6f}  "
                  f"delta_max={float(delta_np.max()):.4f}")

            rows_data.append(dict(
                label=label,
                bicubic_rgb=bgr_to_rgb_u8(bic_bgr),
                alpha_map=alpha_np,
                alpha_mean=a_mean,
                delta_map=delta_np,
                output_rgb=tensor_to_rgb_u8(out_t),
            ))

    out_path = os.path.join(args.output_dir, "gate_heatmap")
    make_figure(rows_data, out_path)

    # Save stats
    stats_path = os.path.join(args.output_dir, "gate_heatmap_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"alpha_vmax used: {max(d['alpha_map'].max() for d in rows_data)*1.1:.6f}\n")
        for d in rows_data:
            f.write(f"{d['label']}: mean={d['alpha_mean']:.6f}  "
                    f"max={float(d['alpha_map'].max()):.6f}\n")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()