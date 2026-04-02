"""
Run this on DGX to show us evaluate_full.py so we can mirror its approach.

cd /raid/home/dgxuser8/capstone1/version1
python read_evaluate_full.py
"""
from pathlib import Path

p = Path('/raid/home/dgxuser8/capstone1/version1/scripts/evaluate_full.py')
print(p.read_text())
