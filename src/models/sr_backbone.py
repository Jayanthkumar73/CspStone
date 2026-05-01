"""
sr_backbone.py — Super-Resolution backbone with FiLM injection.

Uses a lightweight RRDB (Residual-in-Residual Dense Block) architecture
with FiLM layers injected at multiple points for pose conditioning.

Why RRDB over SwinIR:
  - Simpler, fewer parameters, faster training
  - Well-proven for face SR (used in Real-ESRGAN, GFPGAN)
  - Easy to add FiLM layers between blocks
  - Good enough for 112×112 output (we don't need 4K SR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pose_film import FiLMLayer


class DenseBlock(nn.Module):
    """Dense block with 4 convolutions."""

    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        return x4 * 0.2 + x  # residual scaling


class RRDBBlock(nn.Module):
    """Residual-in-Residual Dense Block."""

    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.db1 = DenseBlock(channels, growth)
        self.db2 = DenseBlock(channels, growth)
        self.db3 = DenseBlock(channels, growth)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x


class PoseFiLMSRNet(nn.Module):
    """
    PALF-Net SR backbone: RRDB with FiLM pose conditioning.

    Architecture:
        Input (3, 112, 112) — degraded LR (already upsampled to 112)
        ↓ conv_first (3→64)
        ↓ RRDB block 1 → FiLM(pose) ← pose injection
        ↓ RRDB block 2 → FiLM(pose)
        ↓ RRDB block 3 → FiLM(pose)
        ↓ RRDB block 4 → FiLM(pose)
        ↓ conv_body (64→64) + skip from conv_first
        ↓ conv_up1 (upsample path — optional, for sub-pixel input)
        ↓ conv_last (64→3)
        Output (3, 112, 112) — restored HR face

    The FiLM layers are the NOVELTY: they modulate features based on
    estimated head pose, so the network reconstructs different facial
    structures for frontal vs profile faces.
    """

    def __init__(self, in_channels=3, out_channels=3, num_feat=64,
                 num_block=4, num_grow=32, pose_dim=3, film_hidden=256):
        super().__init__()

        self.num_block = num_block

        # First conv
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        # RRDB blocks
        self.body = nn.ModuleList([
            RRDBBlock(num_feat, num_grow) for _ in range(num_block)
        ])

        # FiLM layers (one after each RRDB block)
        self.film_layers = nn.ModuleList([
            FiLMLayer(num_feat) for _ in range(num_block)
        ])

        # Pose → FiLM params
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, film_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(film_hidden, film_hidden),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gamma_heads = nn.ModuleList([
            nn.Linear(film_hidden, num_feat) for _ in range(num_block)
        ])
        self.beta_heads = nn.ModuleList([
            nn.Linear(film_hidden, num_feat) for _ in range(num_block)
        ])

        # Body conv (after all RRDB blocks)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Output
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

        self.act = nn.LeakyReLU(0.2, inplace=True)

        self._init_film_weights()

    def _init_film_weights(self):
        """Initialize FiLM heads to near-identity (gamma≈0, beta≈0)."""
        for head in list(self.gamma_heads) + list(self.beta_heads):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, pose):
        """
        Args:
            x: (B, 3, 112, 112) — degraded LR face (upsampled to 112)
            pose: (B, 3) — normalized [yaw, pitch, roll] in [-1, 1]
        Returns:
            (B, 3, 112, 112) — restored face
        """
        # Move entire model to x device if needed
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        # Encode pose
        pose = pose.to(x.device)
        pose_feat = self.pose_encoder(pose)

        # First conv
        feat = self.conv_first(x)
        body_feat = feat.clone()  # skip connection

        # RRDB blocks with FiLM conditioning
        for i in range(self.num_block):
            feat = self.body[i](feat)

            # FiLM modulation
            gamma = self.gamma_heads[i](pose_feat)
            beta = self.beta_heads[i](pose_feat)
            feat = self.film_layers[i](feat, gamma, beta)

        # Body conv + global skip
        feat = self.conv_body(feat)
        feat = feat + body_feat

        # Output
        out = self.conv_last(self.act(feat))

        # Residual learning: predict the difference
        out = out + x

        return out


class PoseFiLMSRNetLarge(PoseFiLMSRNet):
    """Larger version for better quality (more blocks, more features)."""

    def __init__(self, **kwargs):
        defaults = dict(num_feat=96, num_block=8, num_grow=48, film_hidden=384)
        defaults.update(kwargs)
        super().__init__(**defaults)
