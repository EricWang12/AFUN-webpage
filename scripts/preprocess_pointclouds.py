#!/usr/bin/env python3
"""
Build lightweight binary point clouds for the in-browser 3D gallery.

For every result folder under static/results/ that has a point_cloud.npz,
this writes pointcloud_small.bin with a downsampled cloud:

  uint32  N
  float32 positions[N, 3]     (XYZ in camera frame, meters)
  uint8   colors[N, 3]        (RGB)

The format is tiny and trivial to read from JavaScript via fetch + ArrayBuffer.
Re-run this script after adding new result folders.

Usage:
  python3 scripts/preprocess_pointclouds.py
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(ROOT, 'static', 'results')

TARGET_POINTS = 10000   # downsample target — ~150 KB per cloud
Z_MIN = 0.05            # discard invalid / behind-camera points
Z_MAX = 8.0             # discard far-away noise
SEED = 42


def downsample(points, colors, target, rng):
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] > Z_MIN)
        & (points[:, 2] < Z_MAX)
    )
    points = points[valid]
    colors = colors[valid]
    n = len(points)
    if n == 0:
        return points, colors
    if n <= target:
        return points, colors
    idx = rng.choice(n, target, replace=False)
    return points[idx], colors[idx]


def write_bin(path, points, colors):
    n = len(points)
    with open(path, 'wb') as f:
        f.write(np.uint32(n).tobytes())
        f.write(points.astype(np.float32, copy=False).tobytes())
        f.write(colors.astype(np.uint8, copy=False).tobytes())


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f'No results dir at {RESULTS_DIR}', file=sys.stderr)
        sys.exit(1)

    rng = np.random.RandomState(SEED)
    total = 0
    for name in sorted(os.listdir(RESULTS_DIR)):
        folder = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(folder):
            continue
        npz_path = os.path.join(folder, 'point_cloud.npz')
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path)
        points = data['points_xyz']
        colors = data['colors_rgb']
        n_in = len(points)

        points, colors = downsample(points, colors, TARGET_POINTS, rng)
        n_out = len(points)

        out_path = os.path.join(folder, 'pointcloud_small.bin')
        write_bin(out_path, points, colors)
        size_kb = os.path.getsize(out_path) / 1024
        print(f'  {name:<35}  {n_in:>7} -> {n_out:>5} pts  ({size_kb:6.1f} KB)')
        total += 1

    print(f'Done. Wrote {total} pointcloud_small.bin files.')


if __name__ == '__main__':
    main()
