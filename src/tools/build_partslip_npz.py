#!/usr/bin/env python3
"""Convert PartSLIP / PartNet-E test.zip into eval-ready NPZ files."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Build PartSLIP NPZ files for evaluation")
    p.add_argument("--partslip_root", type=str, required=True, help="Root with data/test.zip, split/test.txt, PartNetE_meta.json")
    p.add_argument("--output_root", type=str, required=True, help="Output directory for NPZ files")
    p.add_argument("--split", type=str, default="test", help="Dataset split inside zip and split/*.txt")
    p.add_argument("--npoints", type=int, default=2048, help="If >0, deterministically subsample to this many points")
    p.add_argument("--seed", type=int, default=42, help="Base seed for deterministic sampling")
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def deterministic_select(num_points, npoints, seed):
    if npoints <= 0 or npoints == num_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    replace = num_points < npoints
    return np.asarray(rng.choice(num_points, size=npoints, replace=replace), dtype=np.int64)


def read_split_items(split_path):
    items = []
    for line in Path(split_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj_id, category = line.split()
        items.append((obj_id, category))
    return items


def read_partslip_meta(meta_path):
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return {str(cat): [str(x) for x in names] for cat, names in meta.items()}


def parse_ascii_ply_xyz(raw_bytes):
    text = raw_bytes.decode("utf-8", errors="ignore")
    header, body = text.split("end_header\n", 1)
    vertex_count = None
    prop_count = 0
    in_vertex = False
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            in_vertex = True
            prop_count = 0
            continue
        if line.startswith("element ") and not line.startswith("element vertex"):
            in_vertex = False
        if in_vertex and line.startswith("property "):
            prop_count += 1
    if vertex_count is None or prop_count < 3:
        raise RuntimeError("Failed to parse vertex header from PLY")
    arr = np.fromstring(body, sep=" ", dtype=np.float32)
    need = int(vertex_count) * int(prop_count)
    if arr.size < need:
        raise RuntimeError(f"PLY body too short: need {need} floats, found {arr.size}")
    arr = arr[:need].reshape(int(vertex_count), int(prop_count))
    return arr[:, :3].astype(np.float32)


def load_label_payload(raw_bytes):
    payload = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict) or "semantic_seg" not in payload:
        raise RuntimeError("Unexpected PartSLIP label.npy format")
    return np.asarray(payload["semantic_seg"]).reshape(-1).astype(np.int64)


def save_npz(out_path, points, labels, label_names, category, source_key):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points=points.astype(np.float32),
        labels=labels.astype(np.int64),
        label_names=np.asarray(label_names, dtype=object),
        category=np.asarray(category, dtype=object),
        source_key=np.asarray(source_key, dtype=object),
    )


def main():
    args = parse_args()
    root = Path(args.partslip_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    meta = read_partslip_meta(root / "PartNetE_meta.json")
    items = read_split_items(root / "split" / f"{args.split}.txt")
    zip_path = root / "data" / f"{args.split}.zip"
    out_dir = output_root / f"{args.split}_npz_n{args.npoints if args.npoints > 0 else 'full'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for idx, (obj_id, category) in enumerate(items):
            if category not in meta:
                raise RuntimeError(f"Category {category!r} missing from PartNetE_meta.json")
            ply_key = f"{args.split}/{category}/{obj_id}/pc.ply"
            label_key = f"{args.split}/{category}/{obj_id}/label.npy"
            if ply_key not in names or label_key not in names:
                raise RuntimeError(f"Missing archive member for {category}/{obj_id}")

            out_path = out_dir / f"{category}__{obj_id}.npz"
            if out_path.exists() and not args.overwrite:
                continue

            points = parse_ascii_ply_xyz(zf.read(ply_key))
            labels = load_label_payload(zf.read(label_key))
            if labels.shape[0] != points.shape[0]:
                raise RuntimeError(
                    f"Point/label mismatch for {category}/{obj_id}: "
                    f"{points.shape[0]} points vs {labels.shape[0]} labels"
                )

            sample_seed = int(args.seed) + idx
            select = deterministic_select(points.shape[0], int(args.npoints), sample_seed)
            save_npz(
                out_path=out_path,
                points=points[select],
                labels=labels[select],
                label_names=meta[category],
                category=category,
                source_key=f"{args.split}/{category}/{obj_id}",
            )
            written += 1

    print(f"[done] wrote {written} files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
