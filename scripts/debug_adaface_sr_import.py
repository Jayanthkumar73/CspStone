"""
Run on DGX to find the correct AdaFaceSR import path.
cd /raid/home/dgxuser8/capstone1/version1
python debug_adaface_sr_import.py
"""
import sys, os
from pathlib import Path

ROOT = Path('/raid/home/dgxuser8/capstone1/version1')
sys.path.insert(0, str(ROOT))

print("="*60)
print("1. Searching for adaface_sr.py")
print("="*60)
for p in ROOT.rglob('adaface_sr.py'):
    print(f"  FOUND: {p.relative_to(ROOT)}")
    # Show class definition
    for line in p.read_text().split('\n'):
        if 'class AdaFace' in line or 'def __init__' in line or 'def forward' in line:
            print(f"    {line.strip()}")

print()
print("="*60)
print("2. src/ directory structure")
print("="*60)
src = ROOT / 'src'
if src.exists():
    for f in sorted(src.rglob('*.py')):
        print(f"  {f.relative_to(ROOT)}")
else:
    print("  src/ not found")

print()
print("="*60)
print("3. models/ directory structure")
print("="*60)
models = ROOT / 'models'
if models.exists():
    for f in sorted(models.rglob('*.py')):
        print(f"  {f.relative_to(ROOT)}")
else:
    print("  models/ not found")

print()
print("="*60)
print("4. Try imports directly")
print("="*60)
for path_str in ['src.models.adaface_sr', 'models.adaface_sr', 'adaface_sr']:
    try:
        import importlib
        mod = importlib.import_module(path_str)
        print(f"  SUCCESS: {path_str}")
        print(f"    Classes: {[x for x in dir(mod) if not x.startswith('_')]}")
        break
    except Exception as e:
        print(f"  FAIL: {path_str} — {e}")

print()
print("="*60)
print("5. How does evaluate_full.py import it?")
print("="*60)
evf = ROOT / 'scripts/evaluate_full.py'
for line in evf.read_text().split('\n'):
    if 'adaface_sr' in line.lower() or 'AdaFaceSR' in line:
        print(f"  {line.strip()}")
