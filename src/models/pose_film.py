"""
pose_film.py — Feature-wise Linear Modulation conditioned on head pose.

THE CORE NOVELTY OF PALF-NET:
  Standard SR treats all faces the same regardless of viewpoint.
  PALF-Net injects pose information (yaw, pitch, roll) into the SR process
  via FiLM layers, so the network knows what facial structure to reconstruct.

  FiLM: output = gamma(pose) * features + beta(pose)

Phase 0 findings applied:
  - Pose estimation works at 16-48px with SR preprocessing
  - Best preprocessing: GFPGAN (32-48px), esrgan_then_gfpgan (16-24px)
  - Pose normalized to [-1, 1] range (÷90°)
"""

import torch
import torch.nn as nn
import math


class FiLMLayer(nn.Module):
    """Single FiLM conditioning layer."""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # Initialize gamma=1, beta=0 (identity transform at start)
        self.gamma_bias = nn.Parameter(torch.ones(1, feature_dim, 1, 1))
        self.beta_bias = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))

    def forward(self, features, gamma, beta):
        """
        Args:
            features: (B, C, H, W)
            gamma: (B, C) — multiplicative modulation
            beta: (B, C) — additive modulation
        """
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) + self.gamma_bias  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1) + self.beta_bias
        return gamma * features + beta


class PoseFiLMGenerator(nn.Module):
    """
    Maps pose vector → (gamma, beta) for each FiLM layer.

    Input: [yaw, pitch, roll] normalized to [-1, 1]
    Output: gamma and beta for modulating SR features
    """

    def __init__(self, pose_dim=3, feature_dim=64, hidden_dim=256, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.feature_dim = feature_dim

        # Shared pose encoder
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Per-layer gamma/beta predictors
        self.gamma_heads = nn.ModuleList([
            nn.Linear(hidden_dim, feature_dim) for _ in range(n_layers)
        ])
        self.beta_heads = nn.ModuleList([
            nn.Linear(hidden_dim, feature_dim) for _ in range(n_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize so initial output is near identity (gamma≈0, beta≈0)."""
        for head in self.gamma_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.beta_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, pose):
        """
        Args:
            pose: (B, 3) — normalized [yaw, pitch, roll]
        Returns:
            list of (gamma, beta) tuples, one per FiLM layer
        """
        h = self.encoder(pose)
        modulations = []
        for i in range(self.n_layers):
            gamma = self.gamma_heads[i](h)  # (B, feature_dim)
            beta = self.beta_heads[i](h)
            modulations.append((gamma, beta))
        return modulations


class MultiScalePoseFiLM(nn.Module):
    """
    FiLM conditioning at multiple feature scales.
    For encoder-decoder SR architectures with different channel dims at each level.
    """

    def __init__(self, pose_dim=3, feature_dims=(64, 128, 256, 512), hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.film_layers = nn.ModuleList()
        self.gamma_heads = nn.ModuleList()
        self.beta_heads = nn.ModuleList()

        for fd in feature_dims:
            self.film_layers.append(FiLMLayer(fd))
            self.gamma_heads.append(nn.Linear(hidden_dim, fd))
            self.beta_heads.append(nn.Linear(hidden_dim, fd))

        self._init_weights()

    def _init_weights(self):
        for head in list(self.gamma_heads) + list(self.beta_heads):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, feature_list, pose):
        """
        Args:
            feature_list: list of (B, C_i, H_i, W_i) tensors
            pose: (B, 3)
        Returns:
            list of modulated features
        """
        h = self.encoder(pose)
        out = []
        for i, (feat, film) in enumerate(zip(feature_list, self.film_layers)):
            gamma = self.gamma_heads[i](h)
            beta = self.beta_heads[i](h)
            out.append(film(feat, gamma, beta))
        return out
