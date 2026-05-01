"""
degradation.py — Synthetic LR face generation for PALF-Net training.

Simulates real surveillance degradations:
  HR face → blur → downsample → noise → JPEG compression → LR face

Phase 0 findings applied:
  - Target LR range: 8-48px (SCface d1≈15px, d2≈30px, d3≈50px)
  - Training should emphasize 16-32px range (critical surveillance zone)
"""

import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class DegradationPipeline:
    """
    Generate degraded LR face from HR face for training.

    Pipeline: HR → blur → downsample → noise → JPEG → LR
    """

    def __init__(self,
                 lr_range=(8, 48),
                 blur_kernel_range=(3, 9),
                 noise_range=(0, 15),
                 jpeg_range=(30, 95),
                 hr_size=112,
                 weighted_sampling=True):
        """
        Args:
            lr_range: (min_px, max_px) for output LR size
            blur_kernel_range: (min_k, max_k) for Gaussian blur kernel
            noise_range: (min_sigma, max_sigma) for Gaussian noise
            jpeg_range: (min_quality, max_quality) for JPEG compression
            hr_size: target HR size (input and ground truth)
            weighted_sampling: if True, sample more from 16-32px range
        """
        self.lr_min, self.lr_max = lr_range
        self.blur_min, self.blur_max = blur_kernel_range
        self.noise_min, self.noise_max = noise_range
        self.jpeg_min, self.jpeg_max = jpeg_range
        self.hr_size = hr_size
        self.weighted_sampling = weighted_sampling

    def sample_lr_size(self):
        """Sample LR size, biased toward critical surveillance range."""
        if self.weighted_sampling:
            # 60% chance: critical range (16-32), 30% chance: (32-max), 10% chance: (min-16)
            # Focused on the 16-32px zone where surveillance FR is most needed
            r = random.random()
            if r < 0.6:
                return random.randint(max(self.lr_min, 16), min(32, self.lr_max))
            elif r < 0.9:
                return random.randint(min(32, self.lr_max), self.lr_max)
            else:
                return random.randint(self.lr_min, max(self.lr_min, 16))
        else:
            return random.randint(self.lr_min, self.lr_max)

    def __call__(self, hr_img):
        """
        Degrade HR face image.

        Args:
            hr_img: numpy array (H, W, 3) BGR, uint8, should be hr_size × hr_size

        Returns:
            lr_img: degraded LR image resized back to hr_size (numpy, uint8)
            lr_size: actual LR resolution used
            params: dict of degradation parameters applied
        """
        assert hr_img.shape[0] == hr_img.shape[1] == self.hr_size, \
            f"Expected {self.hr_size}×{self.hr_size}, got {hr_img.shape[:2]}"

        img = hr_img.astype(np.float32)
        lr_size = self.sample_lr_size()

        # 1. Gaussian blur (simulate optical defocus)
        blur_k = random.randrange(self.blur_min, self.blur_max + 1, 2)  # must be odd
        blur_sigma = random.uniform(0.5, blur_k / 2)
        img = cv2.GaussianBlur(img, (blur_k, blur_k), blur_sigma)

        # 2. Downsample to LR
        img = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_AREA)

        # 3. Gaussian noise (simulate sensor noise)
        noise_sigma = random.uniform(self.noise_min, self.noise_max)
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
            img = img + noise

        # 4. JPEG compression (simulate storage artifacts)
        jpeg_q = random.randint(self.jpeg_min, self.jpeg_max)
        img = np.clip(img, 0, 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q]
        _, buf = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)

        # 5. Resize back to HR size (this is what the SR network will receive)
        lr_img = cv2.resize(img, (self.hr_size, self.hr_size), interpolation=cv2.INTER_CUBIC)
        lr_img = np.clip(lr_img, 0, 255).astype(np.uint8)

        params = {
            "lr_size": lr_size,
            "blur_k": blur_k,
            "blur_sigma": blur_sigma,
            "noise_sigma": noise_sigma,
            "jpeg_quality": jpeg_q,
        }

        return lr_img, lr_size, params


class DegradationPipelineTorch:
    """
    PyTorch-native version for on-GPU degradation during training.
    Faster but slightly less realistic than OpenCV version.
    """

    def __init__(self, lr_range=(8, 48), noise_range=(0, 15), hr_size=112):
        self.lr_min, self.lr_max = lr_range
        self.noise_min, self.noise_max = noise_range
        self.hr_size = hr_size

    def __call__(self, hr_tensor):
        """
        Args:
            hr_tensor: (B, 3, H, W) float tensor, range [0, 1]
        Returns:
            lr_tensor: (B, 3, H, W) degraded, same size as input
            lr_sizes: list of int, LR sizes used per sample
        """
        B = hr_tensor.shape[0]
        device = hr_tensor.device
        lr_tensors = []
        lr_sizes = []

        for i in range(B):
            img = hr_tensor[i:i+1]  # (1, 3, H, W)

            lr_size = random.randint(self.lr_min, self.lr_max)
            lr_sizes.append(lr_size)

            # Downsample
            lr = F.interpolate(img, size=(lr_size, lr_size), mode='area')

            # Noise
            sigma = random.uniform(self.noise_min, self.noise_max) / 255.0
            if sigma > 0:
                lr = lr + torch.randn_like(lr) * sigma

            # Upsample back
            lr = F.interpolate(lr, size=(self.hr_size, self.hr_size), mode='bicubic',
                               align_corners=False)
            lr = lr.clamp(0, 1)
            lr_tensors.append(lr)

        return torch.cat(lr_tensors, dim=0), lr_sizes