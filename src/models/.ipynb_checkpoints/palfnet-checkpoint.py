"""
palfnet.py — Full PALF-Net pipeline.

Pipeline:
    LR face → [SR preprocess for pose] → 6DRepNet → pose
    LR face + pose → PoseFiLMSRNet → SR face
    SR face → [Frozen ArcFace] → embedding → recognition

This module wraps everything for training and inference.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np

from src.models.sr_backbone import PoseFiLMSRNet
from src.models.identity_loss import PerceptualIdentityLoss


class PALFNet(nn.Module):
    """
    Complete PALF-Net model.

    During training: receives (LR, HR, pose) and optimizes SR quality.
    During inference: receives LR, estimates pose, produces SR output.
    """

    def __init__(self, config=None):
        super().__init__()

        # Defaults
        num_feat = 64
        num_block = 4
        num_grow = 32
        pose_dim = 3
        film_hidden = 256

        if config:
            num_feat = config.get("num_feat", 64)
            num_block = config.get("num_block", 4)
            num_grow = config.get("num_grow", 32)
            pose_dim = config.get("pose_dim", 3)
            film_hidden = config.get("film_hidden", 256)

        # SR backbone with FiLM
        self.sr_net = PoseFiLMSRNet(
            num_feat=num_feat,
            num_block=num_block,
            num_grow=num_grow,
            pose_dim=pose_dim,
            film_hidden=film_hidden,
        )

    def forward(self, lr, pose):
        """
        Args:
            lr: (B, 3, 112, 112) degraded face
            pose: (B, 3) normalized pose [-1, 1]
        Returns:
            sr: (B, 3, 112, 112) restored face
        """
        return self.sr_net(lr, pose)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        film_params = 0
        for name, p in self.named_parameters():
            if 'film' in name or 'gamma' in name or 'beta' in name or 'pose_encoder' in name:
                film_params += p.numel()
        return {
            "total": total,
            "trainable": trainable,
            "film_params": film_params,
            "sr_params": trainable - film_params,
        }


class PALFNetInference:
    """
    Full inference pipeline including pose estimation and SR preprocessing.

    Usage:
        model = PALFNetInference(checkpoint_path, device='cuda')
        sr_face = model.restore(lr_face_bgr)
    """

    def __init__(self, checkpoint_path, device='cuda', pose_sr_method='gfpgan'):
        self.device = device
        self.pose_sr_method = pose_sr_method

        # Load PALF-Net SR model
        self.model = PALFNet()
        ckpt = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(device).eval()

        # Pose estimator (loaded on demand)
        self._pose_est = None
        self._sr_proc = None

    def _get_pose_estimator(self):
        if self._pose_est is None:
            from sixdrepnet import SixDRepNet
            self._pose_est = SixDRepNet()
        return self._pose_est

    def estimate_pose(self, bgr_img):
        """Estimate pose from BGR image, with SR preprocessing for small faces."""
        h, w = bgr_img.shape[:2]
        img = bgr_img

        # Phase 0 finding: SR helps pose at 16-48px
        if max(h, w) < 48:
            img = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

        img = cv2.resize(img, (224, 224))
        pose_est = self._get_pose_estimator()
        try:
            y, p, r = pose_est.predict(img)
            if isinstance(y, (list, np.ndarray)):
                y, p, r = float(y[0]), float(p[0]), float(r[0])
            return np.array([y, p, r], dtype=np.float32)
        except:
            return np.zeros(3, dtype=np.float32)

    @torch.no_grad()
    def restore(self, lr_bgr):
        """
        Full pipeline: LR BGR image → SR BGR image.

        Args:
            lr_bgr: numpy (H, W, 3) BGR uint8
        Returns:
            sr_bgr: numpy (112, 112, 3) BGR uint8
        """
        # Estimate pose
        pose = self.estimate_pose(lr_bgr)
        pose_norm = np.clip(pose / 90.0, -1.0, 1.0)

        # Prepare input
        lr_112 = cv2.resize(lr_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
        lr_rgb = cv2.cvtColor(lr_112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lr_t = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pose_t = torch.from_numpy(pose_norm).unsqueeze(0).to(self.device)

        # SR
        sr_t = self.model(lr_t, pose_t)
        sr_t = sr_t.clamp(0, 1)

        # Convert back
        sr_rgb = sr_t[0].permute(1, 2, 0).cpu().numpy()
        sr_bgr = cv2.cvtColor((sr_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return sr_bgr
