#!/usr/bin/env python3
"""Convert raw FAUST SATR files into PatchAlign3D eval-ready NPZ files."""

import argparse
import json
from pathlib import Path

import numpy as np


COARSE_LABEL_NAMES = ["arm", "head", "leg", "torso"]
FINE_LABEL_NAMES = [
    "arm",
    "head",
    "torso",
    "leg",
    "belly button",
    "nose",
    "eye",
    "chin",
    "mouth",
    "ear",
    "neck",
    "forehead",
    "shoulder",
    "elbow",
    "hand",
    "knee",
    "foot",
]


def parse_args():
    p = argparse.ArgumentParser("Build FAUST NPZ files for evaluation")
    p.add_argument("--faust_root", type=str, required=True, help="Root directory containing scans/ and *_gt.json")
    p.add_argument("--output_root", type=str, required=True, help="Output directory for NPZ files")
    p.add_argument("--label_set", type=str, default="both", choices=["coarse", "fine", "both"])
    p.add_argument("--npoints", type=int, default=2048, help="If >0, deterministically subsample to this many points")
    p.add_argument("--seed", type=int, default=42, help="Base seed for deterministic point sampling")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def load_vertices(obj_path):
    verts = []
    with Path(obj_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise RuntimeError(f"No vertices found in {obj_path}")
    return np.asarray(verts, dtype=np.float32)


def deterministic_select(num_points, npoints, seed):
    if npoints <= 0 or npoints == num_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    replace = num_points < npoints
    return np.asarray(rng.choice(num_points, size=npoints, replace=replace), dtype=np.int64)


def label_mapping(names):
    return {name: idx for idx, name in enumerate(names)}


def save_npz(out_path, points, labels, label_names, source_scan):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points=points.astype(np.float32),
        labels=labels.astype(np.int64),
        label_names=np.asarray(label_names, dtype=object),
        source_scan=str(source_scan),
    )


def convert_label_set(faust_root, output_root, subset_name, label_json_name, label_names, npoints, seed, overwrite):
    label_path = Path(faust_root) / label_json_name
    scans_dir = Path(faust_root) / "scans"
    out_dir = Path(output_root) / f"{subset_name}_npz_n{npoints if npoints > 0 else 'full'}"
    labels_by_scan = json.loads(label_path.read_text(encoding="utf-8"))
    name_to_idx = label_mapping(label_names)
    missing = []
    written = 0

    for mesh_name in sorted(labels_by_scan):
        obj_path = scans_dir / f"{mesh_name}.obj"
        out_path = out_dir / f"{mesh_name}.npz"
        if out_path.exists() and not overwrite:
            continue
        if not obj_path.is_file():
            missing.append(str(obj_path))
            continue

        points = load_vertices(obj_path)
        labels_str = labels_by_scan[mesh_name]
        if len(labels_str) != points.shape[0]:
            raise RuntimeError(
                f"Label/point count mismatch for {mesh_name}: "
                f"{len(labels_str)} labels vs {points.shape[0]} vertices"
            )
        try:
            labels = np.asarray([name_to_idx[name] for name in labels_str], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"Unexpected label {exc.args[0]!r} in {mesh_name}") from exc

        idx = deterministic_select(points.shape[0], npoints, seed + int(mesh_name.split("_")[-1]))
        save_npz(
            out_path=out_path,
            points=points[idx],
            labels=labels[idx],
            label_names=label_names,
            source_scan=obj_path,
        )
        written += 1

    if missing:
        raise RuntimeError(f"Missing {len(missing)} scan files, first missing: {missing[0]}")
    print(f"[{subset_name}] wrote {written} files to {out_dir}", flush=True)
    return out_dir


def main():
    args = parse_args()
    faust_root = Path(args.faust_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    produced = []
    if args.label_set in ("coarse", "both"):
        produced.append(
            convert_label_set(
                faust_root=faust_root,
                output_root=output_root,
                subset_name="coarse",
                label_json_name="coarse_gt.json",
                label_names=COARSE_LABEL_NAMES,
                npoints=int(args.npoints),
                seed=int(args.seed),
                overwrite=args.overwrite,
            )
        )
    if args.label_set in ("fine", "both"):
        produced.append(
            convert_label_set(
                faust_root=faust_root,
                output_root=output_root,
                subset_name="fine",
                label_json_name="fine_grained_gt.json",
                label_names=FINE_LABEL_NAMES,
                npoints=int(args.npoints),
                seed=int(args.seed),
                overwrite=args.overwrite,
            )
        )

    for path in produced:
        print(f"[done] {path}", flush=True)


if __name__ == "__main__":
    main()
