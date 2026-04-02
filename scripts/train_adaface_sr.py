#!/usr/bin/env python3
"""
train_adaface_sr.py — Train AdaFace-SR: Adaptive Identity-Preserving Face SR.

Novel training objective:
    L = λ_id * L_id + λ_pix * L_pix + λ_perc * L_perc + λ_gate * L_gate

Where L_gate is the GATE SPARSITY LOSS:
    L_gate = mean(α)  — penalizes the gate for being open

This forces the model to default to bicubic (α=0) and only open the gate
when doing so actually helps identity preservation.

Usage:
    # Quick test:
    python scripts/train_adaface_sr.py --data_dir data/training/aligned_112 \
        --pose_cache data/training/poses_cache.json \
        --max_samples 1000 --epochs 5 --batch_size 4 --num_workers 0 \
        --exp_name adaface_test

    # Full training:
    nohup python scripts/train_adaface_sr.py \
        --data_dir data/training/aligned_112 \
        --pose_cache data/training/poses_cache.json \
        --max_samples 50000 --epochs 50 --batch_size 8 --num_workers 4 \
        --exp_name adaface_sr_v1 \
        > adaface_log.txt 2>&1 &
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

from src.models.adaface_sr import AdaFaceSR
from src.data.dataset import PALFNetDataset


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


class AdaFaceSRTrainer:
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
        log("Building AdaFace-SR...")
        config = {
            "num_feat": args.num_feat,
            "num_block": args.num_block,
            "num_grow": args.num_grow,
            "res_embed_dim": args.res_embed_dim,
            "gate_channels": args.gate_channels,
        }
        self.model = AdaFaceSR(config).to(self.device)
        info = self.model.count_parameters()
        log(f"  Total params: {info['total']:,}")
        log(f"  SR branch:    {info['sr_params']:,}")
        log(f"  Gate branch:  {info['gate_params']:,}")
        log(f"  Resolution:   {info['res_encoder_params']:,}")

        # Losses
        self.pixel_loss = nn.L1Loss()

        # Differentiable ArcFace identity loss
        self.fr_model = None
        try:
            self._setup_identity_loss()
        except Exception as e:
            log(f"  ⚠️ Identity loss failed: {e}")
            log(f"  Training WITHOUT identity loss (pixel + gate only)")

        # LPIPS perceptual loss
        self.lpips_fn = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device).eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
            log("  ✅ LPIPS perceptual loss ready")
        except:
            log("  ⚠️ LPIPS not available")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr,
            betas=(0.9, 0.999), weight_decay=args.weight_decay)

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        # Dataset
        log(f"\nLoading training data from: {args.data_dir}")
        self.dataset = PALFNetDataset(
            data_root=args.data_dir, hr_size=112, lr_range=(args.lr_min, args.lr_max),
            pose_cache_path=args.pose_cache, augment=True, max_samples=args.max_samples)

        self.dataloader = DataLoader(
            self.dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []

        if args.resume:
            self._load_checkpoint(args.resume)

        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    def _setup_identity_loss(self):
        from src.models.iresnet import load_pretrained_arcface
        self.fr_model = load_pretrained_arcface(
            arch="iresnet50",
            weights_dir=os.path.join(ROOT, "pretrained", "arcface_pytorch")
        ).to(self.device)

        dummy = torch.randn(1, 3, 112, 112).to(self.device)
        out = self.fr_model((dummy - 0.5) / 0.5)
        assert out.shape == (1, 512), f"Unexpected: {out.shape}"
        log(f"  ✅ PyTorch ArcFace ready (differentiable)")

    def compute_loss(self, sr, hr, alpha, lr_sizes, bicubic_input):
        losses = {}

        # Pixel loss
        losses["pixel"] = self.pixel_loss(sr, hr)

        # Perceptual loss
        if self.lpips_fn is not None:
            losses["perceptual"] = self.lpips_fn(sr * 2 - 1, hr * 2 - 1).mean()
        else:
            losses["perceptual"] = torch.tensor(0.0, device=self.device)

        # DIFFERENTIABLE identity loss
        if self.fr_model is not None:
            sr_norm = (sr - 0.5) / 0.5
            emb_sr = self.fr_model(sr_norm)
            with torch.no_grad():
                hr_norm = (hr - 0.5) / 0.5
                emb_hr = self.fr_model(hr_norm)
                # Also get bicubic input embedding for comparison
                bic_norm = (bicubic_input - 0.5) / 0.5
                emb_bic = self.fr_model(bic_norm)

            emb_sr = F.normalize(emb_sr, dim=1)
            emb_hr = F.normalize(emb_hr.detach(), dim=1)
            emb_bic = F.normalize(emb_bic.detach(), dim=1)

            # Standard identity loss: SR output vs HR target
            cos_sr_hr = (emb_sr * emb_hr).sum(dim=1)  # (B,)
            losses["identity"] = (1 - cos_sr_hr).mean()

            # COMPARATIVE IDENTITY LOSS (novel):
            # How does bicubic compare to HR?
            cos_bic_hr = (emb_bic * emb_hr).sum(dim=1)  # (B,)
            # If SR is WORSE than bicubic (cos_sr_hr < cos_bic_hr),
            # penalize proportionally. If SR is BETTER, reward (reduce loss).
            # identity_delta > 0 means SR improved over bicubic
            identity_delta = cos_sr_hr - cos_bic_hr  # (B,)
            # Penalize when SR makes things worse: max(0, -delta) = ReLU(-delta)
            losses["worse_than_bic"] = F.relu(-identity_delta).mean()

        else:
            losses["identity"] = torch.tensor(0.0, device=self.device)
            losses["worse_than_bic"] = torch.tensor(0.0, device=self.device)

        # GATE SPARSITY LOSS
        losses["gate"] = alpha.mean()

        # Weighted total
        total = (self.args.pixel_weight * losses["pixel"]
                 + self.args.perceptual_weight * losses["perceptual"]
                 + self.args.identity_weight * losses["identity"]
                 + self.args.gate_weight * losses["gate"]
                 + self.args.comparative_weight * losses["worse_than_bic"])

        losses["total"] = total
        return losses

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_losses = {"total": 0, "pixel": 0, "perceptual": 0, "identity": 0, "gate": 0, "worse_than_bic": 0}
        n_batches = 0
        total_batches = len(self.dataloader)
        avg_alpha = 0

        for batch_idx, batch in enumerate(self.dataloader):
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            lr_sizes = batch["lr_size"].to(self.device)

            self.optimizer.zero_grad()

            sr, alpha = self.model(lr, lr_sizes)
            sr = sr.clamp(0, 1)

            # lr IS the bicubic-upsampled input (already 112x112)
            losses = self.compute_loss(sr, hr, alpha, lr_sizes, bicubic_input=lr)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()
            avg_alpha += alpha.mean().item()
            n_batches += 1

            pct = 100 * (batch_idx + 1) / total_batches
            avg_loss = epoch_losses["total"] / n_batches
            mean_alpha = avg_alpha / n_batches
            bar_len = 30
            filled = int(bar_len * (batch_idx + 1) / total_batches)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  Epoch {epoch+1}/{self.args.epochs} "
                  f"|{bar}| {pct:5.1f}% "
                  f"[{batch_idx+1}/{total_batches}] "
                  f"loss={avg_loss:.4f} α={mean_alpha:.3f}", end="", flush=True)

        print()
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        avg["mean_alpha"] = avg_alpha / max(n_batches, 1)
        return avg

    def save_samples(self, epoch):
        self.model.eval()
        batch = next(iter(self.dataloader))
        lr = batch["lr"][:4].to(self.device)
        hr = batch["hr"][:4].to(self.device)
        lr_sizes = batch["lr_size"][:4].to(self.device)

        with torch.no_grad():
            sr, alpha = self.model(lr, lr_sizes)
            sr = sr.clamp(0, 1)

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(4):
            for j, (img, title) in enumerate([
                (lr[i], "LR Input"),
                (sr[i], "AdaFace-SR Output"),
                (hr[i], "HR Ground Truth"),
                (alpha[i], f"Gate α (mean={alpha[i].mean():.2f})"),
            ]):
                arr = img.permute(1, 2, 0).cpu().numpy()
                if j == 3:
                    axes[i, j].imshow(arr.mean(axis=2), cmap='hot', vmin=0, vmax=1)
                else:
                    axes[i, j].imshow(arr.clip(0, 1))
                if i == 0:
                    axes[i, j].set_title(title, fontsize=10)
                axes[i, j].axis("off")

        plt.suptitle(f"AdaFace-SR — Epoch {epoch+1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "samples", f"epoch_{epoch+1:03d}.png"), dpi=100)
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
        torch.save(ckpt, os.path.join(self.exp_dir, "checkpoints", "checkpoint_last.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(self.exp_dir, "checkpoints", "checkpoint_best.pth"))
        if (epoch + 1) % self.args.save_every == 0:
            torch.save(ckpt, os.path.join(self.exp_dir, "checkpoints", f"checkpoint_{epoch+1:03d}.pth"))

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
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        for k in ["total", "pixel", "perceptual", "identity", "gate", "worse_than_bic"]:
            vals = [l.get(k, 0) for l in self.train_losses]
            if any(v > 0 for v in vals):
                axes[0].plot(epochs, vals, label=k)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("AdaFace-SR Training Losses")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Gate openness
        alphas = [l.get("mean_alpha", 0) for l in self.train_losses]
        axes[1].plot(epochs, alphas, 'r-', lw=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean Gate Value (α)")
        axes[1].set_title("Gate Openness Over Training")
        axes[1].set_ylim(0, 1)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "plots", "loss_curves.png"), dpi=150)
        plt.close()

    def train(self):
        log(f"\n{'=' * 60}")
        log(f"  AdaFace-SR Training")
        log(f"  Epochs: {self.args.epochs}, Batch: {self.args.batch_size}")
        log(f"  Weights: id={self.args.identity_weight}, pix={self.args.pixel_weight}, "
            f"perc={self.args.perceptual_weight}, gate={self.args.gate_weight}, "
            f"comp={self.args.comparative_weight}")
        log(f"  Device: {self.device}")
        log(f"  Experiment: {self.exp_dir}")
        log(f"{'=' * 60}\n")

        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()
            avg = self.train_one_epoch(epoch)
            self.train_losses.append(avg)
            self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            log(f"Epoch {epoch+1}/{self.args.epochs} ({elapsed:.0f}s) "
                f"loss={avg['total']:.4f} pix={avg['pixel']:.4f} "
                f"id={avg['identity']:.4f} gate={avg['gate']:.4f} "
                f"wtb={avg['worse_than_bic']:.4f} "
                f"α={avg['mean_alpha']:.3f} lr={lr_now:.2e}")

            is_best = avg["total"] < self.best_loss
            if is_best:
                self.best_loss = avg["total"]
            self.save_checkpoint(epoch, is_best)

            if (epoch + 1) % self.args.sample_every == 0 or epoch == 0:
                self.save_samples(epoch)
            if (epoch + 1) % 5 == 0:
                self.plot_losses()

        self.plot_losses()
        log(f"\n{'=' * 60}")
        log(f"  Training complete! Best loss: {self.best_loss:.4f}")
        log(f"  Checkpoints: {self.exp_dir}/checkpoints/")
        log(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Train AdaFace-SR")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pose_cache", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    # Model
    parser.add_argument("--num_feat", type=int, default=64)
    parser.add_argument("--num_block", type=int, default=4)
    parser.add_argument("--num_grow", type=int, default=32)
    parser.add_argument("--res_embed_dim", type=int, default=64)
    parser.add_argument("--gate_channels", type=int, default=32)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_min", type=int, default=8)
    parser.add_argument("--lr_max", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)

    # Loss weights
    parser.add_argument("--pixel_weight", type=float, default=0.1)
    parser.add_argument("--perceptual_weight", type=float, default=0.01)
    parser.add_argument("--identity_weight", type=float, default=5.0)
    parser.add_argument("--gate_weight", type=float, default=1.0,
                        help="Gate sparsity penalty. Higher = more bicubic-like")
    parser.add_argument("--comparative_weight", type=float, default=10.0,
                        help="Penalty for being worse than bicubic input. Key for beating bicubic.")

    # Saving
    parser.add_argument("--exp_name", type=str, default="adaface_sr_latest")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5)

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    trainer = AdaFaceSRTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
