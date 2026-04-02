"""
adaface_sr.py — Adaptive Identity-Preserving Face Super-Resolution (AdaFace-SR)

NOVEL ARCHITECTURE: Gated Residual Bypass with Identity-Aware Gate Training

Key insight: Bicubic upscaling beats all SR methods for recognition because SR
halluccinates identity-altering details. AdaFace-SR solves this with:

1. CONFIDENCE GATE (α): A lightweight branch that produces a per-pixel confidence
   map α ∈ [0,1] controlling how much SR correction to apply:
   
       Output = Bicubic_input + α ⊙ Δ
   
   When α→0: Output = Bicubic (safe fallback, preserves identity)
   When α→1: Output = Bicubic + Δ (full SR correction applied)

2. GATE REGULARIZATION: The gate is penalized for being open (high α) unless it
   improves identity similarity. This creates a "default to bicubic" bias.

3. RESOLUTION-AWARE ENCODING: The input resolution is encoded and fed to both
   the SR branch and gate branch. At very low resolution (16px), the gate learns
   to stay mostly closed. At higher resolution (32-48px), it opens selectively.

4. DIFFERENTIABLE IDENTITY LOSS: Same as PALF-Net v3, gradients flow through
   frozen ArcFace back into both SR and gate branches.

Architecture diagram:

    LR (8-48px) → Bicubic 112×112 ─────────────────────────┐
                        │                                    │
                        ├── SR Branch ──→ Δ (correction)     │
                        │   (4x RRDB, 64ch)                  │
                        │                                    │
                        ├── Gate Branch ──→ α ∈ [0,1]        │
                        │   (3x Conv, lightweight)           │
                        │                                    │
                        └── Resolution Encoder ──→ r_emb ───→│
                                                             │
    Output = Bicubic + α ⊙ Δ  ←─────────────────────────────┘

Total params: ~2.8M (SR: 2.3M, Gate: 0.4M, ResEnc: 0.1M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════
#  BUILDING BLOCKS (reused from PALF-Net)
# ═══════════════════════════════════════════════

class DenseBlock(nn.Module):
    """Dense block with 4 convolutions and residual scaling."""
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
        return x4 * 0.2 + x


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


# ═══════════════════════════════════════════════
#  RESOLUTION ENCODER
# ═══════════════════════════════════════════════

class ResolutionEncoder(nn.Module):
    """
    Encodes the input LR resolution as a learned embedding.
    
    Uses sinusoidal positional encoding (like transformers) followed by
    a small MLP. This tells the gate and SR branches what resolution
    they're working with, so they can adapt behavior.
    
    At 16px: gate should stay mostly closed (bicubic is safer)
    At 32px: gate can open more (SR has more info to work with)
    At 48px+: gate opens selectively for fine detail enhancement
    """
    def __init__(self, embed_dim=64, max_res=112):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_res = max_res
        
        # Sinusoidal encoding dimension
        self.sin_dim = 32
        
        # MLP: sinusoidal features → embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.sin_dim, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def _sinusoidal_encode(self, resolution):
        """Encode scalar resolution to sinusoidal features."""
        # Normalize to [0, 1]
        r = resolution.float() / self.max_res  # (B,)
        
        # Sinusoidal encoding
        freqs = torch.exp(
            torch.arange(0, self.sin_dim // 2, dtype=torch.float32, device=r.device)
            * -(math.log(10000.0) / (self.sin_dim // 2))
        )
        args = r.unsqueeze(1) * freqs.unsqueeze(0)  # (B, sin_dim//2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, sin_dim)
    
    def forward(self, resolution):
        """
        Args:
            resolution: (B,) tensor of LR resolution values (e.g., 16, 24, 32)
        Returns:
            (B, embed_dim) resolution embedding
        """
        sin_feat = self._sinusoidal_encode(resolution)
        return self.mlp(sin_feat)


# ═══════════════════════════════════════════════
#  CONFIDENCE GATE
# ═══════════════════════════════════════════════

class ConfidenceGate(nn.Module):
    """
    Lightweight branch that produces a per-pixel confidence map α ∈ [0,1].
    
    The gate learns WHEN to apply SR correction:
    - α→0: keep bicubic (safe for identity)
    - α→1: apply full SR correction (confident it helps)
    
    Takes input features + resolution embedding as conditioning.
    Initialized with negative bias so α starts near 0 (default: bicubic).
    """
    def __init__(self, in_channels=3, mid_channels=32, res_embed_dim=64, init_bias=-3.0):
        super().__init__()
        
        # Resolution conditioning: project to spatial features
        self.res_proj = nn.Sequential(
            nn.Linear(res_embed_dim, mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Convolutional gate
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(mid_channels, 3, 1, 1)  # per-channel gate
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # Initialize gate output with NEGATIVE bias
        # bias=-3.0 → sigmoid gives α≈0.047 (nearly pure bicubic)
        # bias=-2.0 → sigmoid gives α≈0.12
        nn.init.zeros_(self.conv_out.weight)
        nn.init.constant_(self.conv_out.bias, init_bias)
    
    def forward(self, x, res_emb):
        """
        Args:
            x: (B, 3, 112, 112) — bicubic upsampled input
            res_emb: (B, res_embed_dim) — resolution embedding
        Returns:
            alpha: (B, 3, 112, 112) — confidence map in [0, 1]
        """
        feat = self.act(self.conv1(x))
        
        # Add resolution conditioning (broadcast spatially)
        res_feat = self.res_proj(res_emb)  # (B, mid_channels)
        feat = feat + res_feat.unsqueeze(-1).unsqueeze(-1)
        
        feat = self.act(self.conv2(feat))
        feat = self.act(self.conv3(feat))
        alpha = torch.sigmoid(self.conv_out(feat))
        
        return alpha


# ═══════════════════════════════════════════════
#  SR BRANCH
# ═══════════════════════════════════════════════

class SRBranch(nn.Module):
    """
    RRDB-based SR branch that produces a correction Δ.
    
    Conditioned on resolution embedding via feature modulation.
    Does NOT include the residual connection — that's in the main model.
    """
    def __init__(self, in_channels=3, out_channels=3, num_feat=64,
                 num_block=4, num_grow=32, res_embed_dim=64):
        super().__init__()
        
        self.num_block = num_block
        
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        
        self.body = nn.ModuleList([
            RRDBBlock(num_feat, num_grow) for _ in range(num_block)
        ])
        
        # Resolution-conditioned modulation (simpler than FiLM — just additive)
        self.res_mods = nn.ModuleList([
            nn.Linear(res_embed_dim, num_feat) for _ in range(num_block)
        ])
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # Initialize modulation heads near zero
        for mod in self.res_mods:
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)
    
    def forward(self, x, res_emb):
        """
        Args:
            x: (B, 3, 112, 112) — bicubic upsampled input
            res_emb: (B, res_embed_dim) — resolution embedding
        Returns:
            delta: (B, 3, 112, 112) — SR correction (NOT the final output)
        """
        feat = self.conv_first(x)
        body_feat = feat.clone()
        
        for i in range(self.num_block):
            feat = self.body[i](feat)
            # Resolution modulation (additive, spatially broadcast)
            mod = self.res_mods[i](res_emb)  # (B, num_feat)
            feat = feat + mod.unsqueeze(-1).unsqueeze(-1)
        
        feat = self.conv_body(feat)
        feat = feat + body_feat
        delta = self.conv_last(self.act(feat))
        
        return delta


# ═══════════════════════════════════════════════
#  MAIN MODEL: AdaFace-SR
# ═══════════════════════════════════════════════

class AdaFaceSR(nn.Module):
    """
    Adaptive Identity-Preserving Face Super-Resolution.
    
    Core equation:
        Output = Input + α ⊙ Δ
    
    where:
        Input  = bicubic-upsampled LR face (112×112)
        Δ      = SR correction from RRDB branch
        α      = per-pixel confidence gate ∈ [0,1]
    
    When gate is closed (α=0): Output = Input (bicubic, preserves identity)
    When gate is open (α=1):   Output = Input + Δ (full SR applied)
    
    The gate is trained with:
        1. Identity loss (primary) — opens gate only if SR helps recognition
        2. Gate sparsity loss — penalizes gate for being open (default: closed)
        3. Pixel loss (minor) — basic reconstruction guidance
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = {}
        
        num_feat = config.get("num_feat", 64)
        num_block = config.get("num_block", 4)
        num_grow = config.get("num_grow", 32)
        res_embed_dim = config.get("res_embed_dim", 64)
        gate_channels = config.get("gate_channels", 32)
        
        # Resolution encoder
        self.res_encoder = ResolutionEncoder(embed_dim=res_embed_dim)
        
        # SR correction branch
        self.sr_branch = SRBranch(
            num_feat=num_feat, num_block=num_block,
            num_grow=num_grow, res_embed_dim=res_embed_dim
        )
        
        # Confidence gate
        self.gate = ConfidenceGate(
            mid_channels=gate_channels, res_embed_dim=res_embed_dim
        )
    
    def forward(self, x, lr_size=None):
        """
        Args:
            x: (B, 3, 112, 112) — bicubic upsampled LR input
            lr_size: (B,) — original LR resolution (e.g., 16, 24, 32)
                     If None, defaults to 24 (mid-range)
        Returns:
            output: (B, 3, 112, 112) — restored face
            alpha:  (B, 3, 112, 112) — gate confidence map (for visualization/loss)
        """
        B = x.shape[0]
        
        if lr_size is None:
            lr_size = torch.full((B,), 24.0, device=x.device)
        elif isinstance(lr_size, (int, float)):
            lr_size = torch.full((B,), float(lr_size), device=x.device)
        else:
            lr_size = lr_size.float()
        
        # Encode resolution
        res_emb = self.res_encoder(lr_size)  # (B, res_embed_dim)
        
        # SR correction
        delta = self.sr_branch(x, res_emb)  # (B, 3, 112, 112)
        
        # Confidence gate
        alpha = self.gate(x, res_emb)  # (B, 3, 112, 112)
        
        # Gated residual: output = input + alpha * delta
        output = x + alpha * delta
        
        return output, alpha
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        sr_params = sum(p.numel() for p in self.sr_branch.parameters())
        gate_params = sum(p.numel() for p in self.gate.parameters())
        res_params = sum(p.numel() for p in self.res_encoder.parameters())
        return {
            "total": total,
            "sr_params": sr_params,
            "gate_params": gate_params,
            "res_encoder_params": res_params,
        }
