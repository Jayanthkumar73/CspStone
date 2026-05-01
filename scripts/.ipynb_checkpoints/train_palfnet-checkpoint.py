#!/usr/bin/env python3
"""
train_palfnet.py — Train PALF-Net: Pose-Aware SR for Low-Resolution Face Recognition.

Usage:
    python version1/scripts/train_palfnet.py --data_dir data/training/aligned_112

    # Quick test with small dataset:
    python version1/scripts/train_palfnet.py --data_dir data/training/aligned_112 \
        --max_samples 1000 --epochs 5 --batch_size 4

    # Full training:
    python version1/scripts/train_palfnet.py --data_dir data/training/aligned_112 \
        --epochs 100 --batch_size 8 --lr 1e-4

    # Resume from checkpoint:
    python version1/scripts/train_palfnet.py --data_dir data/training/aligned_112 \
        --resume experiments/palfnet_latest/checkpoint_last.pth
"""

import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.palfnet import PALFNet
from src.models.identity_loss import PerceptualIdentityLoss
from src.data.dataset import PALFNetDataset


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Experiment directory
        self.exp_dir = os.path.join(ROOT, "experiments", args.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)

        # Model
        log("Building PALF-Net...")
        config = {
            "num_feat": args.num_feat,
            "num_block": args.num_block,
            "num_grow": args.num_grow,
            "pose_dim": 3,
            "film_hidden": args.film_hidden,
        }
        self.model = PALFNet(config).to(self.device)
        param_info = self.model.count_parameters()
        log(f"  Total params: {param_info['total']:,}")
        log(f"  FiLM params:  {param_info['film_params']:,} "
            f"({100*param_info['film_params']/param_info['total']:.1f}%)")
        log(f"  SR params:    {param_info['sr_params']:,}")

        # Loss
        self.pixel_loss = nn.L1Loss()
        self.identity_loss_fn = None

        # Try to load identity loss (frozen FR model)
        try:
            self._setup_identity_loss()
        except Exception as e:
            log(f"  ⚠️  Identity loss not available: {e}")
            log(f"  Training with pixel loss only")

        # LPIPS perceptual loss
        self.lpips_fn = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device).eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
            log("  ✅ LPIPS perceptual loss ready")
        except:
            log("  ⚠️  LPIPS not available, using pixel loss only")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,
        )

        # Pose estimator for training data (optional — can use zeros)
        self.pose_estimator = None
        if not args.no_pose:
            try:
                from sixdrepnet import SixDRepNet
                self.pose_estimator = SixDRepNet()
                log("  ✅ 6DRepNet ready (for training pose labels)")
            except:
                log("  ⚠️  Pose estimator not available — using zero poses")

        # Dataset
        log(f"\nLoading training data from: {args.data_dir}")
        self.dataset = PALFNetDataset(
            data_root=args.data_dir,
            hr_size=112,
            lr_range=(args.lr_min, args.lr_max),
            pose_estimator=self.pose_estimator,
            augment=True,
            max_samples=args.max_samples,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []

        # Resume
        if args.resume:
            self._load_checkpoint(args.resume)

        # Save config
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    def _setup_identity_loss(self):
        """Load frozen ArcFace for identity loss."""
        from insightface.app import FaceAnalysis
        insight_dir = os.path.join(ROOT, "pretrained", "recognition")

        # We need the raw model, not the full FaceAnalysis pipeline
        # For now, we'll use a simpler approach: cosine distance on features
        import onnxruntime as ort
        model_path = os.path.join(insight_dir, "models", "buffalo_l", "w600k_r50.onnx")

        if os.path.isfile(model_path):
            # --- ADD THESE 2 LINES TO MUTE WARNINGS ---
            options = ort.SessionOptions()
            options.log_severity_level = 3  # 3 = Error level only (hides warnings)
            
            self.fr_session = ort.InferenceSession(
                model_path,
                sess_options=options, # --- ADD THIS ARGUMENT ---
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

            class ONNXFRWrapper(nn.Module):
                def __init__(self, session):
                    super().__init__()
                    self.session = session

                def forward(self, x):
                    # x: (B, 3, 112, 112) float [0, 1]
                    # ONNX expects (B, 3, 112, 112) float with specific normalization
                    x_np = x.detach().cpu().numpy()
                    x_np = (x_np - 0.5) / 0.5  # normalize to [-1, 1]
                    results = self.session.run(None, {"input.1": x_np.astype(np.float32)})
                    return torch.from_numpy(results[0]).to(x.device)

            self.identity_loss_fn = ONNXFRWrapper(self.fr_session)
            log("  ✅ ArcFace identity loss ready (ONNX)")
        else:
            raise FileNotFoundError(f"FR model not found: {model_path}")

    def compute_loss(self, sr, hr, lr_sizes=None):
        """Compute combined loss."""
        losses = {}

        # L1 pixel loss
        losses["pixel"] = self.pixel_loss(sr, hr)

        # LPIPS perceptual loss
        if self.lpips_fn is not None:
            losses["perceptual"] = self.lpips_fn(sr * 2 - 1, hr * 2 - 1).mean()
        else:
            losses["perceptual"] = torch.tensor(0.0, device=self.device)

        # Identity loss
        if self.identity_loss_fn is not None:
            try:
                with torch.no_grad():
                    emb_sr_np = sr.detach()
                    emb_hr_np = hr.detach()
                    # Compute in eval mode
                    emb_sr = self.identity_loss_fn(emb_sr_np)
                    emb_hr = self.identity_loss_fn(emb_hr_np)
                    emb_sr = F.normalize(emb_sr.float(), dim=1)
                    emb_hr = F.normalize(emb_hr.float(), dim=1)
                    losses["identity"] = (1 - (emb_sr * emb_hr).sum(dim=1)).mean()
            except:
                losses["identity"] = torch.tensor(0.0, device=self.device)
        else:
            losses["identity"] = torch.tensor(0.0, device=self.device)

        # Weighted total
        total = (self.args.pixel_weight * losses["pixel"]
                 + self.args.perceptual_weight * losses["perceptual"]
                 + self.args.identity_weight * losses["identity"])

        losses["total"] = total
        return losses

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_losses = {"pixel": 0, "perceptual": 0, "identity": 0, "total": 0}
        n_batches = 0

        for batch_idx, batch in enumerate(self.dataloader):
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            pose = batch["pose"].to(self.device)

            # Forward
            sr = self.model(lr, pose)
            sr = sr.clamp(0, 1)

            # Loss
            losses = self.compute_loss(sr, hr, batch.get("lr_size"))

            # Backward
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate
            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()
            n_batches += 1

            # Log every N batches
            # Log every N batches (OVERWRITING THE SAME LINE)
            if (batch_idx + 1) % self.args.log_every == 0:
                avg = {k: v / n_batches for k, v in epoch_losses.items()}
                # \r sends cursor to the start of the line, end="" prevents a new line
                sys.stdout.write(
                    f"\r[{time.strftime('%H:%M:%S')}]   Epoch {epoch+1}/{self.args.epochs} "
                    f"[{batch_idx+1}/{len(self.dataloader)}] "
                    f"L_total={avg['total']:.4f} "
                    f"L_pix={avg['pixel']:.4f} "
                    f"L_perc={avg['perceptual']:.4f} "
                    f"L_id={avg['identity']:.4f}"
                )
                sys.stdout.flush() # Force the terminal to update immediately

        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        return avg

    def save_samples(self, epoch):
        """Save visual comparison: LR | SR | HR."""
        self.model.eval()
        batch = next(iter(self.dataloader))
        lr = batch["lr"][:4].to(self.device)
        hr = batch["hr"][:4].to(self.device)
        pose = batch["pose"][:4].to(self.device)

        with torch.no_grad():
            sr = self.model(lr, pose).clamp(0, 1)

        fig, axes = plt.subplots(4, 3, figsize=(9, 12))
        for i in range(4):
            for j, (img, title) in enumerate([
                (lr[i], "LR (degraded)"),
                (sr[i], "SR (PALF-Net)"),
                (hr[i], "HR (ground truth)"),
            ]):
                arr = img.permute(1, 2, 0).cpu().numpy()
                axes[i, j].imshow(arr)
                if i == 0:
                    axes[i, j].set_title(title)
                axes[i, j].axis("off")

            # Show pose
            p = pose[i].cpu().numpy() * 90
            axes[i, 0].set_ylabel(f"y={p[0]:.0f}° p={p[1]:.0f}°", fontsize=8)

        plt.suptitle(f"Epoch {epoch+1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "samples", f"epoch_{epoch+1:03d}.png"),
                    dpi=100)
        plt.close()

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "config": vars(self.args),
        }

        # Always save latest
        path = os.path.join(self.exp_dir, "checkpoints", "checkpoint_last.pth")
        torch.save(ckpt, path)

        # Save best
        if is_best:
            path = os.path.join(self.exp_dir, "checkpoints", "checkpoint_best.pth")
            torch.save(ckpt, path)

        # Save periodic
        if (epoch + 1) % self.args.save_every == 0:
            path = os.path.join(self.exp_dir, "checkpoints", f"checkpoint_{epoch+1:03d}.pth")
            torch.save(ckpt, path)

    def _load_checkpoint(self, path):
        log(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_loss = ckpt.get("best_loss", float('inf'))
        self.train_losses = ckpt.get("train_losses", [])
        log(f"  Resumed at epoch {self.start_epoch}, best_loss={self.best_loss:.4f}")

    def plot_losses(self):
        if not self.train_losses:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = range(1, len(self.train_losses) + 1)

        keys = ["total", "pixel", "perceptual", "identity"]
        for k in keys:
            vals = [l.get(k, 0) for l in self.train_losses]
            if any(v > 0 for v in vals):
                ax.plot(epochs, vals, label=k)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("PALF-Net Training Losses")
        ax.legend()
        ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "plots", "loss_curves.png"), dpi=150)
        plt.close()

    def train(self):
        log(f"\n{'=' * 60}")
        log(f"  Starting training: {self.args.epochs} epochs")
        log(f"  Batch size: {self.args.batch_size}")
        log(f"  LR range: {self.args.lr_min}-{self.args.lr_max}px")
        log(f"  Device: {self.device}")
        log(f"  Experiment: {self.exp_dir}")
        log(f"{'=' * 60}\n")

        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()

            # Train
            avg_losses = self.train_one_epoch(epoch)
            self.train_losses.append(avg_losses)

            # LR step
            self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0

            log(f"Epoch {epoch+1}/{self.args.epochs} "
                f"({elapsed:.0f}s) "
                f"loss={avg_losses['total']:.4f} "
                f"pix={avg_losses['pixel']:.4f} "
                f"perc={avg_losses['perceptual']:.4f} "
                f"id={avg_losses['identity']:.4f} "
                f"lr={lr_now:.2e}")

            # Save
            is_best = avg_losses["total"] < self.best_loss
            if is_best:
                self.best_loss = avg_losses["total"]
            self.save_checkpoint(epoch, is_best)

            # Samples
            if (epoch + 1) % self.args.sample_every == 0 or epoch == 0:
                self.save_samples(epoch)

            # Plot
            if (epoch + 1) % 5 == 0:
                self.plot_losses()

        # Final
        self.plot_losses()
        log(f"\n{'=' * 60}")
        log(f"  Training complete!")
        log(f"  Best loss: {self.best_loss:.4f}")
        log(f"  Checkpoints: {self.exp_dir}/checkpoints/")
        log(f"  Samples:     {self.exp_dir}/samples/")
        log(f"{'=' * 60}")


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train PALF-Net")

    # Data
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to training images (aligned 112×112 faces)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (for quick testing)")

    # Model
    parser.add_argument("--num_feat", type=int, default=64)
    parser.add_argument("--num_block", type=int, default=4)
    parser.add_argument("--num_grow", type=int, default=32)
    parser.add_argument("--film_hidden", type=int, default=256)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_min", type=int, default=8, help="Min LR size for degradation")
    parser.add_argument("--lr_max", type=int, default=48, help="Max LR size for degradation")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pose", action="store_true",
                        help="Skip pose estimation (use zeros)")

    # Loss weights
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--identity_weight", type=float, default=0.5)

    # Saving
    parser.add_argument("--exp_name", type=str, default="palfnet_latest")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
