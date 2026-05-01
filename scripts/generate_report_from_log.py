#!/usr/bin/env python3
"""Parse qmul_tea_run3.log and generate a clean final report."""
import re, sys

log_file = sys.argv[1] if len(sys.argv) > 1 else 'qmul_tea_run3.log'

with open(log_file) as f:
    content = f.read()

# Parse single-frame Rank-1
single = {}
pattern_single = r'\[(\d{2}:\d{2}:\d{2})\]\s+([\w\-\s()]+?)\s+\|\s+([\d.]+)%.*\|\s+([\d.]+)%'
for m in re.finditer(pattern_single, content):
    pass  # fallback: use TEA table

# Parse the TEA summary table printed at end
tea_pattern = r'([\w\-].*?)\s*\|\s+([\d.]+)%\s*\|\s+([\d.]+)%\s*\|\s+\+([\d.]+)pp'
results = []
for m in re.finditer(tea_pattern, content):
    method = m.group(1).strip()
    sf = float(m.group(2))
    tea = float(m.group(3))
    delta = float(m.group(4))
    results.append((method, sf, tea, delta))

# Also parse native rank-1 from per-method lines
native_pattern = r'(\w[\w\-\s()]+?)\s*Rank-1=([\d.]+)%'
native = {}
for m in re.finditer(native_pattern, content):
    native[m.group(1).strip()] = float(m.group(2))

print('=' * 70)
print(' QMUL-SurvFace Evaluation Report — Run 3 (Fixed)')
print('=' * 70)
print(f'{"Method":<25} | {"Single-Frame":>14} | {"TEA Rank-1":>10} | {"Improvement":>12}')
print('-' * 70)
for method, sf, tea, delta in results:
    print(f'{method:<25} | {sf:>13.1f}% | {tea:>9.1f}% | +{delta:>10.2f}pp')
print('=' * 70)

# Highlight best
if results:
    best = max(results, key=lambda x: x[2])
    print(f'\n>>> BEST METHOD: {best[0]} with TEA Rank-1 = {best[2]:.1f}%')
    bicubic = next((r for r in results if 'Bicubic' in r[0]), None)
    if bicubic:
        gain = best[2] - bicubic[2]
        print(f'>>> Improvement over Bicubic baseline: +{gain:.2f}pp')

print()
print('TEA = Temporal Evidence Aggregation (quality-weighted multi-frame fusion)')
print('All results on QMUL-SurvFace (60,423 probe images, 3,000 gallery subjects)')
