#!/usr/bin/env python3
"""
organize_scface.py — Organize raw SCface dump into expected structure.

Usage:
    python version1/scripts/organize_scface.py
    python version1/scripts/organize_scface.py --raw_dir /path/to/scface
"""

import os, re, shutil, json, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCFACE_DIR = os.path.join(ROOT, "data", "scface")


def find_images(d):
    imgs = []
    for dp, _, fns in os.walk(d):
        for f in fns:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                imgs.append(os.path.join(dp, f))
    return imgs


def categorize(raw_dir):
    cats = {'mugshot': [], 'd1': [], 'd2': [], 'd3': [], 'unknown': []}

    for img in find_images(raw_dir):
        p = img.lower()
        if any(x in p for x in ['mugshot', 'frontal', 'gallery', 'hr']):
            cats['mugshot'].append(img)
        elif any(x in p for x in ['distance1', 'dist1', '/d1/', '_d1', '_1.']):
            cats['d1'].append(img)
        elif any(x in p for x in ['distance2', 'dist2', '/d2/', '_d2', '_2.']):
            cats['d2'].append(img)
        elif any(x in p for x in ['distance3', 'dist3', '/d3/', '_d3', '_3.']):
            cats['d3'].append(img)
        else:
            # Try filename pattern: 001_cam1_d1
            m = re.match(r'(\d{3})_\w*(\d)_\w*(\d)', os.path.basename(img))
            if m:
                dist = int(m.group(3))
                if dist in [1, 2, 3]:
                    cats[f'd{dist}'].append(img)
                else:
                    cats['unknown'].append(img)
            elif re.match(r'^\d{3}\.\w+$', os.path.basename(img)):
                cats['mugshot'].append(img)
            else:
                cats['unknown'].append(img)
    return cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=SCFACE_DIR)
    args = parser.parse_args()

    print("=" * 50)
    print("  SCface Organizer")
    print("=" * 50)

    all_imgs = find_images(args.raw_dir)
    print(f"\n  Found {len(all_imgs)} images in {args.raw_dir}")
    if not all_imgs:
        print("  ❌ No images. Unzip SCface here first.")
        return

    cats = categorize(args.raw_dir)
    print(f"\n  Mugshots:   {len(cats['mugshot']):>5}")
    print(f"  Distance 1: {len(cats['d1']):>5}  (~4.2m, ~15×15 px)")
    print(f"  Distance 2: {len(cats['d2']):>5}  (~2.6m, ~30×30 px)")
    print(f"  Distance 3: {len(cats['d3']):>5}  (~1.0m, ~50×50 px)")
    print(f"  Unknown:    {len(cats['unknown']):>5}")

    out = os.path.join(SCFACE_DIR, "organized")
    dest_map = {
        'mugshot': os.path.join(out, 'mugshot'),
        'd1': os.path.join(out, 'surveillance', 'd1'),
        'd2': os.path.join(out, 'surveillance', 'd2'),
        'd3': os.path.join(out, 'surveillance', 'd3'),
    }

    for cat, dest in dest_map.items():
        os.makedirs(dest, exist_ok=True)
        for src in cats[cat]:
            shutil.copy2(src, os.path.join(dest, os.path.basename(src)))

    meta = {k: len(v) for k, v in cats.items()}
    with open(os.path.join(out, 'dataset_info.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✅ Organized into: {out}/")
    print(f"    mugshot/         ({meta['mugshot']})")
    print(f"    surveillance/d1/ ({meta['d1']})")
    print(f"    surveillance/d2/ ({meta['d2']})")
    print(f"    surveillance/d3/ ({meta['d3']})")


if __name__ == "__main__":
    main()
