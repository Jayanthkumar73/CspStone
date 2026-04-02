"""
Check what keys the AdaFace IR-101 checkpoint actually has vs iresnet100.
cd /raid/home/dgxuser8/capstone1/version1
python debug_ir101_keys.py
"""
import sys, torch
from pathlib import Path
ROOT = Path('/raid/home/dgxuser8/capstone1/version1')
sys.path.insert(0, str(ROOT))

ckpt_path = ROOT / 'pretrained/recognition/adaface_ir101_webface4m.ckpt'
ckpt  = torch.load(str(ckpt_path), map_location='cpu')

print("Top-level keys:", list(ckpt.keys()))
state = ckpt.get('state_dict', ckpt)
keys  = list(state.keys())
print(f"\nTotal checkpoint keys: {len(keys)}")
print("First 10 keys:")
for k in keys[:10]:
    print(f"  {k}: {state[k].shape}")

print("\nKeys containing 'body':", [k for k in keys if 'body' in k][:5])
print("Keys containing 'layer':", [k for k in keys if 'layer' in k][:5])
print("Keys containing 'input':", [k for k in keys if 'input' in k][:5])
print("Keys containing 'output':", [k for k in keys if 'output' in k][:5])
print("Keys containing 'bn':", [k for k in keys if '.bn' in k][:5])

# Check iresnet100 keys
from src.models.iresnet import iresnet100
model = iresnet100()
model_keys = list(model.state_dict().keys())
print(f"\niresnet100 model keys: {len(model_keys)}")
print("First 10:")
for k in model_keys[:10]:
    print(f"  {k}")

# Find overlap
ckpt_set  = set(k.replace('model.', '') for k in keys if not k.startswith('head.'))
model_set = set(model_keys)
overlap   = ckpt_set & model_set
print(f"\nOverlap: {len(overlap)}/{len(model_set)} model keys found in checkpoint")
print("Missing from checkpoint:", list(model_set - ckpt_set)[:5])
print("Extra in checkpoint:", list(ckpt_set - model_set)[:5])
