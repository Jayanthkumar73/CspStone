"""
Run on DGX:
  cd /raid/home/dgxuser8/capstone1/version1
  python debug3_evalfull.py 2>&1 | tee debug3_output.txt
"""
from pathlib import Path

BASE = Path('/raid/home/dgxuser8/capstone1/version1')

# 1. Show evaluate_full.py in full
print("="*60)
print("evaluate_full.py")
print("="*60)
evf = BASE / 'scripts/evaluate_full.py'
print(evf.read_text())

# 2. Show surveillance directory structure
print("\n" + "="*60)
print("surveillance/ structure")
print("="*60)
surv = BASE / 'data/scface/organized/surveillance'
for item in sorted(surv.iterdir()):
    if item.is_dir():
        files = list(item.glob('*'))
        print(f"  {item.name}/  ({len(files)} files)")
        for f in sorted(files)[:3]:
            print(f"    {f.name}")

# 3. Show models/ directory
print("\n" + "="*60)
print("models/ directory")
print("="*60)
for f in sorted((BASE / 'models').glob('*.py')):
    print(f"  {f.name}")

# 4. Show how adaface_sr is imported in any existing script
print("\n" + "="*60)
print("How AdaFace-SR is imported in existing scripts")
print("="*60)
for f in BASE.rglob('*.py'):
    txt = f.read_text()
    if 'AdaFaceSR' in txt or 'adaface_sr' in txt:
        for i, line in enumerate(txt.split('\n')):
            if 'AdaFaceSR' in line or 'adaface_sr' in line or 'sys.path' in line:
                print(f"  {f.relative_to(BASE)}:{i+1}: {line.strip()}")

# 5. Mugshot naming — how subject ID extracted
print("\n" + "="*60)
print("Mugshot naming pattern (first 5 frontal)")
print("="*60)
mug = BASE / 'data/scface/organized/mugshot'
frontals = sorted(mug.glob('*frontal*'))[:5]
for f in frontals:
    print(f"  {f.name}")
