#!/usr/bin/env python3
"""
precompute_poses.py — Estimate pose for all training images and save to JSON.

Run ONCE before training. Then train_palfnet.py reads poses from file
instead of computing them live (which was 100x slower).

Usage:
    python scripts/precompute_poses.py --data_dir data/training/aligned_112
    python scripts/precompute_poses.py --data_dir data/training/aligned_112 --max_samples 50000

Output:
    data/training/poses_cache.json
"""

import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch_display", type=int, default=500)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.data_dir.rstrip("/")), "poses_cache.json")

    log("=" * 60)
    log("  Pose Precomputation for PALF-Net Training")
    log("=" * 60)

    # Find all images
    image_paths = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    for dp, _, fns in os.walk(args.data_dir):
        for f in fns:
            if os.path.splitext(f)[1].lower() in exts:
                image_paths.append(os.path.join(dp, f))
    image_paths.sort()

    if args.max_samples and len(image_paths) > args.max_samples:
        import random
        random.seed(42)
        random.shuffle(image_paths)
        image_paths = image_paths[:args.max_samples]
        image_paths.sort()

    log(f"  Images found: {len(image_paths)}")

    # Load pose estimator
    log("Loading 6DRepNet...")
    from sixdrepnet import SixDRepNet
    pose_model = SixDRepNet()
    log("  ✅ Ready")

    # Process all images
    poses = {}
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            poses[img_path] = [0.0, 0.0, 0.0]
            n_fail += 1
            continue

        # Resize to 224 for pose estimation
        img_224 = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        try:
            y, p, r = pose_model.predict(img_224)
            if isinstance(y, (list, np.ndarray)):
                y = float(np.asarray(y).flatten()[0])
            if isinstance(p, (list, np.ndarray)):
                p = float(np.asarray(p).flatten()[0])
            if isinstance(r, (list, np.ndarray)):
                r = float(np.asarray(r).flatten()[0])
            poses[img_path] = [float(y), float(p), float(r)]
            n_ok += 1
        except:
            poses[img_path] = [0.0, 0.0, 0.0]
            n_fail += 1

        if (i + 1) % args.batch_display == 0 or i == len(image_paths) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(image_paths) - i - 1) / rate
            pct = 100 * (i + 1) / len(image_paths)
            print(f"\r  [{pct:5.1f}%] {i+1}/{len(image_paths)} "
                  f"ok={n_ok} fail={n_fail} "
                  f"rate={rate:.1f} img/s ETA={eta/60:.0f}min", end="", flush=True)

    print()

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(poses, f)

    elapsed = time.time() - t0
    log(f"\n  Done in {elapsed/60:.1f} minutes")
    log(f"  Poses saved: {args.output}")
    log(f"  Total: {len(poses)}, OK: {n_ok}, Failed: {n_fail}")

    # Stats
    yaws = [v[0] for v in poses.values() if v != [0.0, 0.0, 0.0]]
    if yaws:
        log(f"\n  Yaw distribution:")
        log(f"    Mean: {np.mean(yaws):.1f}°")
        log(f"    Std:  {np.std(yaws):.1f}°")
        log(f"    Min:  {np.min(yaws):.1f}°  Max: {np.max(yaws):.1f}°")
        frontal = sum(1 for y in yaws if abs(y) < 10)
        moderate = sum(1 for y in yaws if 10 <= abs(y) < 25)
        oblique = sum(1 for y in yaws if 25 <= abs(y) < 45)
        profile = sum(1 for y in yaws if abs(y) >= 45)
        log(f"    Frontal (0-10°):  {frontal} ({100*frontal/len(yaws):.1f}%)")
        log(f"    Moderate (10-25°): {moderate} ({100*moderate/len(yaws):.1f}%)")
        log(f"    Oblique (25-45°): {oblique} ({100*oblique/len(yaws):.1f}%)")
        log(f"    Profile (45-90°): {profile} ({100*profile/len(yaws):.1f}%)")


if __name__ == "__main__":
    main()
