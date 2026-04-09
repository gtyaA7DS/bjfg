from __future__ import annotations

import argparse
import re
import signal
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from render_point_cloud import load_point_cloud, render_point_cloud


DEFAULT_ARCHIVE = Path("/root/data/core/core_part-00001.tar.zst")
DEFAULT_OUTPUT_DIR = Path("/root/bjfg_new/bjfg/outputs/core_tar_renders")
ID_PATTERN = re.compile(r"^labeled/points/([^/]+)/points\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a few point clouds one by one from core_part tar.zst and render them."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to the .tar.zst archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where rendered images will be saved.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="How many point clouds to render.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset in the archive id list.",
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
        "--use-input-colors",
        action="store_true",
        help="Use colors stored in the point cloud. Disabled by default.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout


def list_point_ids(archive: Path, limit: int | None = None) -> list[str]:
    ids: list[str] = []
    process = subprocess.Popen(
        ["tar", "--zstd", "-tf", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None

    try:
        for line in process.stdout:
            match = ID_PATTERN.match(line.strip())
            if match:
                ids.append(match.group(1))
                if limit is not None and len(ids) >= limit:
                    process.send_signal(signal.SIGTERM)
                    break
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.stdout:
            process.stdout.close()

    if not ids:
        raise RuntimeError(f"No point cloud ids found in archive: {archive}")
    return ids


def extract_id_files(archive: Path, point_id: str, destination: Path) -> Path:
    members = [
        f"labeled/points/{point_id}/points.pt",
        f"labeled/points/{point_id}/rgb.pt",
    ]
    run_command(["tar", "--zstd", "-xf", str(archive), "-C", str(destination), *members])
    extracted_dir = destination / "labeled" / "points" / point_id
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Failed to extract point cloud directory for id {point_id}")
    return extracted_dir


def render_ids(
    archive: Path,
    output_dir: Path,
    point_ids: list[str],
    point_size: float,
    elev: float,
    azim: float,
    use_input_colors: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    with TemporaryDirectory(prefix="core_tar_render_") as temp_dir:
        temp_root = Path(temp_dir)
        for point_id in point_ids:
            work_root = temp_root / point_id
            work_root.mkdir(parents=True, exist_ok=True)
            extracted_dir = extract_id_files(archive, point_id, work_root)
            points, colors, _ = load_point_cloud(extracted_dir)
            if not use_input_colors:
                colors = None
            output_path = output_dir / f"{point_id}.png"
            render_point_cloud(
                points=points,
                colors=colors,
                output_path=output_path,
                point_size=point_size,
                elev=elev,
                azim=azim,
                roll=0.0,
                title=point_id,
            )
            saved_paths.append(output_path)
            shutil.rmtree(work_root)

    return saved_paths


def main() -> None:
    args = parse_args()
    required = args.start + args.count
    all_ids = list_point_ids(args.archive, limit=required)
    selected_ids = all_ids[args.start : args.start + args.count]
    if not selected_ids:
        raise ValueError("No ids selected. Adjust --start or --count.")

    saved_paths = render_ids(
        archive=args.archive,
        output_dir=args.output_dir,
        point_ids=selected_ids,
        point_size=args.point_size,
        elev=args.elev,
        azim=args.azim,
        use_input_colors=args.use_input_colors,
    )

    print(f"Selected {len(selected_ids)} ids from {args.archive}")
    for point_id, path in zip(selected_ids, saved_paths):
        print(f"{point_id} -> {path}")


if __name__ == "__main__":
    main()
