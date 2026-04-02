"""
identity_loss.py — Identity-preserving loss using frozen FR backbone.

L_identity = 1 - cosine_similarity(FR(SR_face), FR(HR_face))

Ensures the SR output preserves identity information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    """
    Identity loss using a frozen face recognition model.
    Computes cosine distance between embeddings of SR and HR faces.
    """

    def __init__(self, fr_model):
        """
        Args:
            fr_model: nn.Module that takes (B, 3, 112, 112) → (B, 512) embeddings
        """
        super().__init__()
        self.fr_model = fr_model
        for p in self.fr_model.parameters():
            p.requires_grad = False
        self.fr_model.eval()

    @torch.no_grad()
    def forward(self, sr_face, hr_face):
        """
        Args:
            sr_face: (B, 3, 112, 112) range [0, 1]
            hr_face: (B, 3, 112, 112) range [0, 1]
        Returns:
            identity loss scalar (lower = better)
        """
        emb_sr = self.fr_model(sr_face)
        emb_hr = self.fr_model(hr_face)
        emb_sr = F.normalize(emb_sr, dim=1)
        emb_hr = F.normalize(emb_hr, dim=1)
        cos_sim = (emb_sr * emb_hr).sum(dim=1)
        return (1 - cos_sim).mean()


class PerceptualIdentityLoss(nn.Module):
    """
    Combined perceptual + identity loss.
    Uses VGG features for texture quality + FR embeddings for identity.
    """

    def __init__(self, fr_model=None, perceptual_weight=0.1, identity_weight=0.5):
        super().__init__()
        self.pw = perceptual_weight
        self.iw = identity_weight
        self.identity_loss = IdentityLoss(fr_model) if fr_model else None

        # Simple perceptual: L1 on VGG features
        # We'll use LPIPS if available, else skip
        self.lpips = None
        try:
            import lpips
            self.lpips = lpips.LPIPS(net='alex', verbose=False)
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
        except ImportError:
            pass

    def forward(self, sr, hr):
        losses = {}

        # Pixel loss
        losses["pixel"] = F.l1_loss(sr, hr)

        # Perceptual loss
        if self.lpips is not None:
            # LPIPS expects [-1, 1]
            losses["perceptual"] = self.lpips(sr * 2 - 1, hr * 2 - 1).mean()
        else:
            losses["perceptual"] = torch.tensor(0.0, device=sr.device)

        # Identity loss
        if self.identity_loss is not None:
            losses["identity"] = self.identity_loss(sr, hr)
        else:
            losses["identity"] = torch.tensor(0.0, device=sr.device)

        total = (losses["pixel"]
                 + self.pw * losses["perceptual"]
                 + self.iw * losses["identity"])

        losses["total"] = total
        return losses
