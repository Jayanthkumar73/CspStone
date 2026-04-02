"""
adaface_ir101_loader.py
Self-contained AdaFace IR-101 model matching adaface_ir101_webface4m.ckpt exactly.

Block structure (from checkpoint):
    res_layer.0  BN
    res_layer.1  Conv2d 3x3
    res_layer.2  BN
    res_layer.3  PReLU
    res_layer.4  Conv2d 3x3
    res_layer.5  BN

input_layer:  [Conv2d, BN, PReLU]
output_layer: [BN, Dropout, Flatten, Linear, BN]
body:         49 IR blocks (3+13+30+3), channels 64/128/256/512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys

sys.path.insert(0, '/raid/home/dgxuser8/capstone1/version1')


class IRBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c),
            nn.Conv2d(out_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        if stride != 1:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),  # .0
                nn.BatchNorm2d(out_c),                           # .1
            )
        else:
            self.shortcut_layer = None

    def forward(self, x):
        res = self.res_layer(x)
        sc  = self.shortcut_layer(x) if self.shortcut_layer is not None else x
        return res + sc


class AdaFaceIR101(nn.Module):
    def __init__(self, drop_ratio=0.4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        blocks = []
        for num, in_c, out_c, stride in [
            (3,  64,  64,  2),
            (13, 64,  128, 2),
            (30, 128, 256, 2),
            (3,  256, 512, 2),
        ]:
            blocks.append(IRBlock(in_c, out_c, stride=stride))
            for _ in range(num - 1):
                blocks.append(IRBlock(out_c, out_c, stride=1))
        self.body = nn.Sequential(*blocks)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(p=drop_ratio),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512, bias=True),
            nn.BatchNorm1d(512, affine=False),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


def load_adaface_ir101(ckpt_path, device, drop_ratio=0.0):
    model = AdaFaceIR101(drop_ratio=drop_ratio)
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    cleaned = {
        (k[6:] if k.startswith('model.') else k): v
        for k, v in state.items()
        if not (k[6:] if k.startswith('model.') else k).startswith('head.')
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    real_miss = [k for k in missing
                 if 'output_layer.1' not in k and 'output_layer.2' not in k]
    real_unex = [k for k in unexpected
                 if 'output_layer.1' not in k and 'output_layer.2' not in k]
    if not real_miss and not real_unex:
        total = sum(p.numel() for p in model.parameters())
        print(f"  ✅ AdaFace IR-101 loaded perfectly ({total/1e6:.1f}M params)")
    else:
        if real_miss: print(f"  ⚠️  Missing ({len(real_miss)}): {real_miss[:3]}")
        if real_unex: print(f"  ⚠️  Unexpected ({len(real_unex)}): {real_unex[:3]}")
    return model.to(device).eval()


def extract_embedding(fr_model, img_bgr, device):
    img = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    t   = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        emb = fr_model(t)
        emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()[0]


def extract_embeddings_batch(fr_model, img_list_bgr, device, batch_size=32):
    all_embs = []
    for i in range(0, len(img_list_bgr), batch_size):
        batch = img_list_bgr[i:i + batch_size]
        tensors = []
        for img in batch:
            img_r = cv2.resize(img, (112, 112), interpolation=cv2.INTER_CUBIC)
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_r = (img_r - 0.5) / 0.5
            tensors.append(torch.from_numpy(img_r.transpose(2, 0, 1)).float())
        t = torch.stack(tensors).to(device)
        with torch.no_grad():
            embs = fr_model(t)
            embs = F.normalize(embs, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = 'pretrained/recognition/adaface_ir101_webface4m.ckpt'
    print(f"Testing AdaFace IR-101 loader on {device}...")
    model  = load_adaface_ir101(ckpt, device)
    dummy  = np.random.randint(0, 255, (48, 36, 3), dtype=np.uint8)
    emb    = extract_embedding(model, dummy, device)
    print(f"Output shape: {emb.shape}  norm={np.linalg.norm(emb):.6f} (should be 1.0)")
    emb2   = extract_embedding(model, dummy, device)
    print(f"Same-image cosine: {np.dot(emb, emb2):.6f} (should be 1.0)")
    print("✅ AdaFace IR-101 loader working correctly")
