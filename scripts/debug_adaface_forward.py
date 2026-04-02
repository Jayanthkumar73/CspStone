"""
Check AdaFaceSR forward pass output exactly.
cd /raid/home/dgxuser8/capstone1/version1
python debug_adaface_forward.py
"""
import sys, torch
from pathlib import Path
ROOT = Path('/raid/home/dgxuser8/capstone1/version1')
sys.path.insert(0, str(ROOT))

from src.models.adaface_sr import AdaFaceSR
import torch

# Load from checkpoint
ckpt_path = ROOT / 'experiments/adaface_sr_v2/checkpoints/checkpoint_best.pth'
ckpt = torch.load(str(ckpt_path), map_location='cpu')

print("Checkpoint keys:", list(ckpt.keys()))
print("Config:", ckpt.get('config', 'NOT FOUND'))

cfg = ckpt.get('config', {})
model = AdaFaceSR(cfg)
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state, strict=False)
model.eval()

# Test forward pass
x = torch.randn(1, 3, 112, 112)
r = torch.tensor([16.0])

with torch.no_grad():
    out = model(x, r)

print(f"\nOutput type: {type(out)}")
if isinstance(out, (tuple, list)):
    print(f"Tuple length: {len(out)}")
    for i, o in enumerate(out):
        print(f"  out[{i}]: shape={o.shape}, range=[{o.min():.3f}, {o.max():.3f}]")
else:
    print(f"Single tensor: shape={out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")

# Also test without r
with torch.no_grad():
    out2 = model(x)
print(f"\nWithout r — output type: {type(out2)}")
if isinstance(out2, (tuple, list)):
    print(f"  out[0] range: [{out2[0].min():.3f}, {out2[0].max():.3f}]")
else:
    print(f"  range: [{out2.min():.3f}, {out2.max():.3f}]")
