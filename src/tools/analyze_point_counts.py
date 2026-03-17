#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser("Analyze point counts under labeled/points")
    p.add_argument("--data_root", type=str, required=True, help="Root like /root/data/core_subset")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of files to scan (0 means all)")
    return p.parse_args()


def main():
    args = parse_args()
    points_root = Path(args.data_root) / "labeled" / "points"
    if not points_root.exists():
        raise FileNotFoundError(f"Points directory not found: {points_root}")

    files = sorted(points_root.glob("*/points.pt"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No points.pt files found under {points_root}")

    counts = []
    for path in files:
        pts = torch.load(path, map_location="cpu")
        if not isinstance(pts, torch.Tensor):
            raise TypeError(f"Expected tensor in {path}, got {type(pts)}")
        if pts.ndim == 2:
            n = int(pts.shape[0])
        else:
            n = int(pts.reshape(-1, pts.shape[-1]).shape[0])
        counts.append(n)

    counts_sorted = sorted(counts)
    total = len(counts_sorted)

    def q(pct):
        if total == 1:
            return counts_sorted[0]
        idx = round((pct / 100.0) * (total - 1))
        return counts_sorted[idx]

    print(f"scanned_files={total}")
    print(f"min_points={counts_sorted[0]}")
    print(f"p25_points={q(25)}")
    print(f"median_points={q(50)}")
    print(f"p75_points={q(75)}")
    print(f"max_points={counts_sorted[-1]}")
    print(f"mean_points={sum(counts_sorted) / total:.2f}")

    freq = {}
    for n in counts:
        freq[n] = freq.get(n, 0) + 1
    print("top_counts=")
    for n, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
        print(f"  points={n} count={c}")


if __name__ == "__main__":
    main()
