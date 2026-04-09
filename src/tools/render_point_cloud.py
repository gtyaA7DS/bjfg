from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_INPUT = Path("/root/data/core/labeled/points/000074a334c541878360457c672b6c2e")
DEFAULT_OUTPUT = Path("/root/bjfg_new/bjfg/outputs/point_cloud_render.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a point cloud from .pcd or .pt and save a 3D render image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Point cloud file or directory. Directory mode prefers points_with_normals_rgb_face.pcd.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the rendered PNG image.",
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        default=None,
        help="Optional RGB tensor path when --input points to a .pt point tensor.",
    )
    parser.add_argument(
        "--use-input-colors",
        action="store_true",
        help="Use colors stored in the input point cloud. Disabled by default.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=10.0,
        help="Point size for matplotlib scatter.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=22.0,
        help="Elevation angle in degrees.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=38.0,
        help="Azimuth angle in degrees.",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Roll angle in degrees.",
    )
    return parser.parse_args()


def resolve_input(input_path: Path) -> tuple[Path, Path | None]:
    if input_path.is_dir():
        pcd_path = input_path / "points_with_normals_rgb_face.pcd"
        points_path = input_path / "points.pt"
        rgb_path = input_path / "rgb.pt"
        if pcd_path.exists():
            return pcd_path, None
        if points_path.exists():
            return points_path, rgb_path if rgb_path.exists() else None
        raise FileNotFoundError(f"No supported point cloud found in directory: {input_path}")
    return input_path, None


def load_ascii_pcd(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    fields: list[str] | None = None
    data_lines: list[str] = []
    data_mode: str | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("FIELDS "):
                fields = line.split()[1:]
            elif upper.startswith("DATA "):
                data_mode = line.split()[1].lower()
                if data_mode != "ascii":
                    raise ValueError(f"Only ASCII PCD is supported, got DATA {data_mode}")
                break

        if fields is None:
            raise ValueError(f"PCD header in {path} is missing FIELDS")
        if data_mode != "ascii":
            raise ValueError(f"PCD header in {path} is missing ASCII DATA declaration")

        for raw_line in f:
            line = raw_line.strip()
            if line:
                data_lines.append(line)

    data = np.array([[float(v) for v in line.split()] for line in data_lines], dtype=np.float32)
    field_index = {name: idx for idx, name in enumerate(fields)}

    required = ["x", "y", "z"]
    missing = [name for name in required if name not in field_index]
    if missing:
        raise ValueError(f"PCD file {path} is missing required fields: {missing}")

    xyz = data[:, [field_index["x"], field_index["y"], field_index["z"]]]

    rgb = None
    if all(name in field_index for name in ("r", "g", "b")):
        rgb = data[:, [field_index["r"], field_index["g"], field_index["b"]]]

    return xyz, rgb


def load_pt_points(path: Path, rgb_path: Path | None) -> tuple[np.ndarray, np.ndarray | None]:
    points = torch.load(path, map_location="cpu")
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(points)!r}")

    rgb = None
    candidate_rgb = rgb_path
    if candidate_rgb is None and path.name == "points.pt":
        sibling_rgb = path.with_name("rgb.pt")
        if sibling_rgb.exists():
            candidate_rgb = sibling_rgb
    if candidate_rgb is not None and candidate_rgb.exists():
        rgb_tensor = torch.load(candidate_rgb, map_location="cpu")
        if isinstance(rgb_tensor, torch.Tensor):
            rgb = rgb_tensor

    return points.detach().cpu().numpy(), None if rgb is None else rgb.detach().cpu().numpy()


def load_point_cloud(input_path: Path, rgb_path: Path | None = None) -> tuple[np.ndarray, np.ndarray | None, Path]:
    resolved_input, implied_rgb = resolve_input(input_path)
    rgb_candidate = rgb_path if rgb_path is not None else implied_rgb

    if resolved_input.suffix.lower() == ".pcd":
        points, colors = load_ascii_pcd(resolved_input)
    elif resolved_input.suffix.lower() == ".pt":
        points, colors = load_pt_points(resolved_input, rgb_candidate)
    else:
        raise ValueError(f"Unsupported input format: {resolved_input}")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected point tensor/array with shape [N, 3+], got {points.shape}")

    return points[:, :3].astype(np.float32), normalize_colors(colors), resolved_input


def normalize_colors(colors: np.ndarray | None) -> np.ndarray | None:
    if colors is None:
        return None
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] < 3:
        return None
    colors = colors[:, :3]
    if colors.max() > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def compute_limits(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    limits = tuple((center[i] - radius, center[i] + radius) for i in range(3))
    return limits


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    output_path: Path,
    point_size: float,
    elev: float,
    azim: float,
    roll: float,
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=240)
    ax = fig.add_subplot(111, projection="3d")

    if colors is None:
        color_values = np.full((len(points), 3), (0.18, 0.2, 0.24), dtype=np.float32)
        cmap = None
    else:
        color_values = colors
        cmap = None

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color_values,
        cmap=cmap,
        s=point_size,
        depthshade=False,
        linewidths=0,
    )

    limits = compute_limits(points)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, pad=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    points, colors, resolved_input = load_point_cloud(args.input, args.rgb)
    if not args.use_input_colors:
        colors = None
    render_point_cloud(
        points=points,
        colors=colors,
        output_path=args.output,
        point_size=args.point_size,
        elev=args.elev,
        azim=args.azim,
        roll=args.roll,
        title=resolved_input.stem,
    )
    print(f"Rendered {len(points)} points from {resolved_input}")
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
