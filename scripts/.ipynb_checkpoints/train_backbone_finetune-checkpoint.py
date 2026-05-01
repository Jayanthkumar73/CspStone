"""
Backbone Fine-Tuning on QMUL-SurvFace
Fine-tunes iResNet-50 (ArcFace) on real surveillance images from QMUL training_set.
After this, re-run eval_qmul_verification.py with the fine-tuned backbone.

Usage:
    python scripts/train_backbone_finetune.py \
        --data_dir data/QMUL-SurvFace/training_set \
        --backbone pretrained/arcface_pytorch/iresnet50.pth \
        --epochs 30 \
        --exp_name backbone_qmul_v1 \
        > backbone_finetune_log.txt 2>&1
"""

import os, sys, time, argparse, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, '/raid/home/dgxuser8/capstone1/version1')

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',    required=True,  help='QMUL training_set path')
parser.add_argument('--backbone',    required=True,  help='iResNet-50 pretrained checkpoint')
parser.add_argument('--epochs',      type=int, default=30)
parser.add_argument('--batch_size',  type=int, default=64)
parser.add_argument('--lr',          type=float, default=1e-4)
parser.add_argument('--max_samples', type=int, default=0, help='0=all')
parser.add_argument('--exp_name',    default='backbone_qmul_v1')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
exp_dir = f'/raid/home/dgxuser8/capstone1/version1/experiments/{args.exp_name}'
os.makedirs(f'{exp_dir}/checkpoints', exist_ok=True)

print(f"\n{'='*60}")
print(f"  Backbone Fine-Tuning on QMUL Surveillance Data")
print(f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}")
print(f"  Experiment: {exp_dir}")
print(f"{'='*60}\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
class QMULDataset(Dataset):
    """
    QMUL training_set: directories named by identity ID,
    each containing [PersonID]_[CameraID]_[ImageName].jpg files.
    Applies random degradation augmentation to simulate surveillance conditions.
    """
    def __init__(self, data_dir, max_samples=0):
        self.samples  = []  # (path, label)
        self.id2label = {}

        identity_dirs = sorted(os.listdir(data_dir))
        print(f"  Scanning {len(identity_dirs)} identity directories...")

        for label, id_dir in enumerate(identity_dirs):
            id_path = os.path.join(data_dir, id_dir)
            if not os.path.isdir(id_path):
                continue
            self.id2label[id_dir] = label
            for fname in os.listdir(id_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(id_path, fname), label))

        if max_samples > 0 and len(self.samples) > max_samples:
            idx = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in idx]

        self.num_classes = len(identity_dirs)
        print(f"  Dataset: {len(self.samples)} images, {self.num_classes} identities")

    def __len__(self):
        return len(self.samples)

    def augment(self, img):
        """Light augmentation — keep degradation realistic, don't over-augment."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        # Random brightness/contrast (mild)
        alpha = np.random.uniform(0.8, 1.2)
        beta  = np.random.uniform(-15, 15)
        img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # Return blank image if load fails
            img = np.zeros((112, 112, 3), dtype=np.uint8)

        # Resize to 112x112
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
        img = self.augment(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        t   = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return t, label


# ── ArcFace loss head ─────────────────────────────────────────────────────────
class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Loss head.
    s=64, m=0.5 — standard ArcFace hyperparameters.
    """
    def __init__(self, feat_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s          = s
        self.m          = m
        self.weight     = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

        import math
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine   = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi    = cosine * self.cos_m - sine * self.sin_m  # cos(θ+m)
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        logits  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits  = logits * self.s
        return F.cross_entropy(logits, labels.long())


# ── Load backbone ─────────────────────────────────────────────────────────────
from src.models.iresnet import iresnet50

print("[1/4] Loading backbone...")
backbone = iresnet50()
state    = torch.load(args.backbone, map_location='cpu')
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
backbone.load_state_dict(state, strict=False)
backbone = backbone.to(device)
print(f"  ✅ iResNet-50 loaded from {args.backbone}")

# ── Dataset & loader ──────────────────────────────────────────────────────────
print("\n[2/4] Loading QMUL dataset...")
dataset = QMULDataset(args.data_dir, max_samples=args.max_samples)
loader  = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)
print(f"  Steps per epoch: {len(loader)}")

# ── ArcFace head ──────────────────────────────────────────────────────────────
print("\n[3/4] Building ArcFace classification head...")
head = ArcFaceHead(feat_dim=512, num_classes=dataset.num_classes).to(device)
print(f"  Head: 512 → {dataset.num_classes} classes")

# ── Optimizer — lower LR for backbone, higher for head ────────────────────────
optimizer = AdamW([
    {'params': backbone.parameters(), 'lr': args.lr,        'name': 'backbone'},
    {'params': head.parameters(),     'lr': args.lr * 10.0, 'name': 'head'},
], weight_decay=5e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

# ── Training ──────────────────────────────────────────────────────────────────
print(f"\n[4/4] Training for {args.epochs} epochs...")
print(f"{'='*60}")

best_loss = float('inf')

for epoch in range(args.epochs):
    backbone.train()
    head.train()

    epoch_loss = 0.0
    correct    = 0
    total      = 0
    t0         = time.time()

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        # Forward
        features = backbone(imgs)          # (B, 512)
        loss     = head(features, labels)

        # Accuracy (approx — cosine similarity against weight matrix)
        with torch.no_grad():
            logits  = F.linear(F.normalize(features),
                               F.normalize(head.weight)) * head.s
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5.0)
        optimizer.step()

        epoch_loss += loss.item()

        # Progress bar
        pct  = (step + 1) / len(loader)
        bar  = '█' * int(pct * 30) + '░' * (30 - int(pct * 30))
        print(f"\r  Epoch {epoch+1}/{args.epochs} |{bar}| "
              f"{pct*100:5.1f}% [{step+1}/{len(loader)}] "
              f"loss={loss.item():.4f}", end='', flush=True)

    print()  # newline after progress bar

    avg_loss = epoch_loss / len(loader)
    acc      = correct / total * 100
    elapsed  = time.time() - t0
    lr_bb    = optimizer.param_groups[0]['lr']

    print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1}/{args.epochs} "
          f"({elapsed:.0f}s) loss={avg_loss:.4f} acc={acc:.1f}% lr={lr_bb:.2e}")

    scheduler.step()

    # Save best checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        ckpt = {
            'epoch':      epoch + 1,
            'best_loss':  best_loss,
            'backbone_state_dict': backbone.state_dict(),
            'head_state_dict':     head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(args),
        }
        torch.save(ckpt, f'{exp_dir}/checkpoints/checkpoint_best.pth')
        print(f"  ✅ Best checkpoint saved (loss={best_loss:.4f})")

    # Save periodic checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(ckpt, f'{exp_dir}/checkpoints/checkpoint_{epoch+1:03d}.pth')

print(f"\n{'='*60}")
print(f"  Training complete! Best loss: {best_loss:.4f}")
print(f"  Backbone saved to: {exp_dir}/checkpoints/checkpoint_best.pth")
print(f"\n  Next step — extract fine-tuned backbone weights:")
print(f"  python3 -c \"")
print(f"  import torch")
print(f"  ckpt = torch.load('{exp_dir}/checkpoints/checkpoint_best.pth')")
print(f"  torch.save(ckpt['backbone_state_dict'],")
print(f"             '{exp_dir}/checkpoints/backbone_only.pth')\"")
print(f"\n  Then re-run verification eval with --backbone_ckpt pointing to backbone_only.pth")
print(f"{'='*60}\n")
