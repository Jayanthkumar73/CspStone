"""
Run on DGX:
  cd /raid/home/dgxuser8/capstone1/version1
  python debug_backbone_qmul.py
"""
import sys, os
from pathlib import Path

ROOT = Path('/raid/home/dgxuser8/capstone1/version1')
sys.path.insert(0, str(ROOT))

# ── 1. QMUL image paths ──────────────────────────────────────────────────────
print("="*60)
print("1. QMUL image directory structure")
print("="*60)
qmul = ROOT / 'data/QMUL-SurvFace'
for item in sorted(qmul.iterdir()):
    if item.is_dir():
        n = sum(1 for _ in item.rglob('*') if _.is_file())
        print(f"  {item.name}/  ({n} files)")
    else:
        print(f"  {item.name}")

# Check where images actually are
print()
print("Searching for .jpg files under QMUL-SurvFace...")
jpg_dirs = set()
for p in list(qmul.rglob('*.jpg'))[:10]:
    jpg_dirs.add(str(p.parent.relative_to(qmul)))
    print(f"  {p.relative_to(qmul)}")
print(f"  Unique parent dirs: {jpg_dirs}")

# Show what the .mat files reference
print()
print("First 3 positive pair filenames from .mat:")
import scipy.io
pos_mat = qmul / 'Face_Verification_Test_Set/positive_pairs_names.mat'
data = scipy.io.loadmat(str(pos_mat))
key = [k for k in data if not k.startswith('_')][0]
raw = data[key]
for i in range(3):
    a = str(raw[i,0][0]) if raw[i,0].size > 0 else ''
    b = str(raw[i,1][0]) if raw[i,1].size > 0 else ''
    print(f"  pair {i}: '{a}'  '{b}'")
    # Try to find these files
    for fname in [a, b]:
        fname_only = Path(fname).name
        matches = list(qmul.rglob(fname_only))
        if matches:
            print(f"    FOUND: {matches[0].relative_to(qmul)}")
        else:
            print(f"    NOT FOUND: {fname_only}")

# ── 2. iResNet-50 import ─────────────────────────────────────────────────────
print()
print("="*60)
print("2. iResNet-50 import attempts")
print("="*60)
for mod_name in ['backbones', 'src.backbones', 'backbones.iresnet',
                 'src.models.iresnet', 'iresnet']:
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        attrs = [x for x in dir(mod) if 'resnet' in x.lower() or 'iresnet' in x.lower()]
        print(f"  SUCCESS: {mod_name}  attrs={attrs}")
    except Exception as e:
        print(f"  FAIL: {mod_name} — {e}")

# Check src/models/iresnet.py exists
iresnet_file = ROOT / 'src/models/iresnet.py'
if iresnet_file.exists():
    print(f"  src/models/iresnet.py EXISTS")
    for line in iresnet_file.read_text().split('\n'):
        if 'def iresnet' in line or 'class IResNet' in line:
            print(f"    {line.strip()}")

# ── 3. AdaFace IR-101 import ─────────────────────────────────────────────────
print()
print("="*60)
print("3. AdaFace IR-101 import attempts")
print("="*60)
for mod_name in ['src.models.net', 'net', 'models.net',
                 'src.models.iresnet', 'backbones']:
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        attrs = [x for x in dir(mod) if 'build' in x.lower() or 'ir_101' in x.lower()
                 or 'iresnet100' in x.lower()]
        if attrs:
            print(f"  SUCCESS: {mod_name}  attrs={attrs}")
        else:
            print(f"  imported {mod_name} but no relevant attrs")
    except Exception as e:
        print(f"  FAIL: {mod_name} — {e}")
