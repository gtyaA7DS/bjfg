from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise RuntimeError("open3d is required for mesh reconstruction and rendering.") from exc

try:
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise RuntimeError("scipy is required for nearest-face assignment.") from exc

try:  # Optional: prefer PyTorch3D when available.
    from pytorch3d.renderer import (
        FoVOrthographicCameras,
        MeshRasterizer,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    PYTORCH3D_AVAILABLE = True
except Exception:
    PYTORCH3D_AVAILABLE = False


DEFAULT_DATA_ROOT = Path("/root/data/core_subset")
DEFAULT_POINT_SOURCE_ROOT = Path("/root/data/core/labeled/points")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render multi-view RGB images and pix2face maps for PatchAlign3D/Find3D-style "
            "DINO patch feature precomputation."
        )
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--point_source_root",
        type=Path,
        default=DEFAULT_POINT_SOURCE_ROOT,
        help="Fallback directory for rgb/normals/points when the subset only contains points.pt.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--item_ids",
        nargs="*",
        default=None,
        help="Explicit item ids such as flip-flop_(sandal)_000074...; overrides --split.",
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=500)
    parser.add_argument("--camera_dist", type=float, default=3.0)
    parser.add_argument("--fov_deg", type=float, default=35.0)
    parser.add_argument("--poisson_depth", type=int, default=8)
    parser.add_argument("--density_quantile", type=float, default=0.02)
    parser.add_argument("--target_faces", type=int, default=16000)
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "pytorch3d", "open3d"],
        help="PyTorch3D is preferred when available; Open3D ray casting is the fallback.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_mesh", action="store_true")
    return parser.parse_args()


def read_split(root: Path, split: str) -> list[str]:
    split_path = root / "labeled" / "split" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def select_item_ids(args: argparse.Namespace) -> list[str]:
    if args.item_ids:
        return list(args.item_ids)
    items = read_split(args.data_root, args.split)
    return items[args.start : args.start + args.count]


def uid_from_item_id(item_id: str) -> str:
    return item_id.split("_")[-1]


def load_first_available(paths: Iterable[Path]) -> torch.Tensor | None:
    for path in paths:
        if path.exists():
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(path, map_location="cpu")
    return None


def load_points_metadata(data_root: Path, point_source_root: Path, uid: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    subset_dir = data_root / "labeled" / "points" / uid
    source_dir = point_source_root / uid
    point_candidates = [subset_dir / "points.pt", source_dir / "points.pt"]
    points_tensor = load_first_available(point_candidates)
    if points_tensor is None:
        raise FileNotFoundError(f"Could not find points.pt for uid={uid} in {subset_dir} or {source_dir}")

    rgb_tensor = load_first_available([subset_dir / "rgb.pt", source_dir / "rgb.pt"])
    normals_tensor = load_first_available([subset_dir / "normals.pt", source_dir / "normals.pt"])

    points = np.asarray(points_tensor, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points with shape [N, 3+], got {points.shape}")
    points = points[:, :3]

    colors = None
    if rgb_tensor is not None:
        colors = np.asarray(rgb_tensor, dtype=np.float32)
        if colors.ndim == 2 and colors.shape[1] >= 3:
            colors = colors[:, :3]
            if colors.max() > 1.0:
                colors = colors / 255.0
            colors = np.clip(colors, 0.0, 1.0)
        else:
            colors = None

    normals = None
    if normals_tensor is not None:
        normals = np.asarray(normals_tensor, dtype=np.float32)
        if normals.ndim == 2 and normals.shape[1] >= 3:
            normals = normals[:, :3]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.clip(norms, 1e-8, None)
        else:
            normals = None

    return points, colors, normals


def normalize_xyz(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = xyz.mean(axis=0, keepdims=True)
    centered = xyz - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    scale = max(scale, 1e-6)
    return centered / scale, center.reshape(3), scale


def build_pseudo_colors(points: np.ndarray, normals: np.ndarray | None = None) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:3]
        proj = centered @ basis.T
    except np.linalg.LinAlgError:
        proj = centered

    span = np.max(np.abs(proj), axis=0, keepdims=True)
    proj = proj / np.clip(span, 1e-6, None)
    colors = 0.5 + 0.5 * proj[:, :3]

    if normals is not None and normals.shape == points.shape:
        normal_rgb = 0.5 + 0.5 * normals[:, :3]
        colors = 0.7 * colors + 0.3 * normal_rgb

    colors = np.clip(colors, 0.0, 1.0)
    return colors.astype(np.float32)


def build_open3d_point_cloud(points: np.ndarray, colors: np.ndarray | None, normals: np.ndarray | None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is None:
        colors = build_pseudo_colors(points, normals)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if normals is not None and normals.shape == points.shape:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    else:
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(min(50, max(10, points.shape[0] // 100)))
    return pcd


def reconstruct_mesh(
    points: np.ndarray,
    colors: np.ndarray | None,
    normals: np.ndarray | None,
    poisson_depth: int,
    density_quantile: float,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pcd = build_open3d_point_cloud(points, colors, normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(poisson_depth),
        n_threads=0,
    )
    densities = np.asarray(densities)
    if densities.size > 0 and 0.0 < density_quantile < 1.0:
        threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if target_faces > 0 and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        raise RuntimeError("Mesh reconstruction failed: no vertices or faces produced.")

    point_tree = cKDTree(points)
    nearest_point_idx = point_tree.query(verts, k=1)[1]
    if colors is None:
        point_colors = build_pseudo_colors(points, normals)
        vertex_colors = point_colors[nearest_point_idx].astype(np.float32)
    else:
        vertex_colors = colors[nearest_point_idx].astype(np.float32)

    return verts, faces, vertex_colors


def assign_points_to_faces(points: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_centroids = verts[faces].mean(axis=1)
    face_tree = cKDTree(face_centroids)
    point2face = face_tree.query(points, k=1)[1].astype(np.int64)
    return point2face


def spherical_eye(dist: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    x = dist * math.cos(elev) * math.sin(azim)
    y = dist * math.sin(elev)
    z = dist * math.cos(elev) * math.cos(azim)
    return np.array([x, y, z], dtype=np.float32)


def get_elev_azim_sequences(num_views: int) -> tuple[np.ndarray, np.ndarray]:
    azim = np.linspace(-180.0, 180.0, num=num_views, endpoint=False, dtype=np.float32)
    base_elev = np.array([30.0, -20.0], dtype=np.float32)
    elev = np.tile(base_elev, int(np.ceil(num_views / len(base_elev))))[:num_views]
    return elev, azim


def render_with_open3d(
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    num_views: int,
    image_size: int,
    camera_dist: float,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(verts.astype(np.float32)),
        o3d.core.Tensor(faces.astype(np.int32)),
    )
    mesh.vertex["colors"] = o3d.core.Tensor(vertex_colors.astype(np.float32))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    invalid_id = np.iinfo(np.uint32).max
    images: list[np.ndarray] = []
    pix2face: list[np.ndarray] = []
    elev_seq, azim_seq = get_elev_azim_sequences(num_views)
    background = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    for elev, azim in zip(elev_seq, azim_seq):
        eye = spherical_eye(camera_dist, float(elev), float(azim))
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=float(fov_deg),
            center=o3d.core.Tensor([0.0, 0.0, 0.0], dtype=o3d.core.float32),
            eye=o3d.core.Tensor(eye, dtype=o3d.core.float32),
            up=o3d.core.Tensor([0.0, 1.0, 0.0], dtype=o3d.core.float32),
            width_px=int(image_size),
            height_px=int(image_size),
        )
        ans = scene.cast_rays(rays)
        prim_ids = ans["primitive_ids"].numpy().reshape(image_size, image_size)
        uvs = ans["primitive_uvs"].numpy().reshape(image_size, image_size, 2).astype(np.float32)

        face_map = prim_ids.astype(np.int64)
        valid = prim_ids != invalid_id
        face_map[~valid] = -1

        image = background.copy()
        if valid.any():
            valid_faces = faces[face_map[valid]]
            uv = uvs[valid]
            w0 = (1.0 - uv[:, 0] - uv[:, 1])[:, None]
            w1 = uv[:, 0:1]
            w2 = uv[:, 1:2]
            colors = (
                vertex_colors[valid_faces[:, 0]] * w0
                + vertex_colors[valid_faces[:, 1]] * w1
                + vertex_colors[valid_faces[:, 2]] * w2
            )
            image[valid] = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)

        images.append(image)
        pix2face.append(face_map)

    return np.stack(images, axis=0), np.stack(pix2face, axis=0)


def render_with_pytorch3d(
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    num_views: int,
    image_size: int,
    camera_dist: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if not PYTORCH3D_AVAILABLE:
        raise RuntimeError("PyTorch3D is not available in this environment.")

    verts_t = torch.from_numpy(verts).float().to(device)
    faces_t = torch.from_numpy(faces).long().to(device)
    colors_t = torch.from_numpy(vertex_colors).float().to(device)

    elev, azim = get_elev_azim_sequences(num_views)
    R, T = look_at_view_transform(dist=float(camera_dist), elev=elev.tolist(), azim=azim.tolist(), device=device)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=int(image_size),
        blur_radius=1e-5,
        faces_per_pixel=5,
        bin_size=0,
        perspective_correct=False,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

    mesh = Meshes(
        verts=[verts_t],
        faces=[faces_t],
        textures=TexturesVertex(verts_features=[colors_t]),
    )
    mesh_b = mesh.extend(num_views)
    fragments = rasterizer(mesh_b, cameras=cameras)
    images = shader(fragments, mesh_b, cameras=cameras, lights=lights)[..., :3]

    pix = fragments.pix_to_face[..., 0]
    n_faces = faces.shape[0]
    valid = pix >= 0
    pix = torch.where(valid, pix % n_faces, torch.full_like(pix, -1))
    rgb = (images.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    return rgb, pix.long().cpu().numpy()


def choose_backend(args: argparse.Namespace) -> str:
    if args.backend == "auto":
        return "pytorch3d" if PYTORCH3D_AVAILABLE else "open3d"
    if args.backend == "pytorch3d" and not PYTORCH3D_AVAILABLE:
        raise RuntimeError("Requested backend=pytorch3d, but pytorch3d is not installed.")
    return args.backend


def save_outputs(
    data_root: Path,
    item_id: str,
    uid: str,
    images: np.ndarray,
    pix2face: np.ndarray,
    point2face: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    overwrite: bool,
    save_mesh: bool,
    backend: str,
) -> None:
    render_dir = data_root / "labeled" / "rendered" / item_id / "oriented"
    img_dir = render_dir / "imgs"
    point_dir = data_root / "labeled" / "points" / uid
    pix_path = render_dir / "pix2face.pt"
    p2f_path = point_dir / "point2face.pt"

    if render_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Render directory already exists: {render_dir}. Pass --overwrite to replace.")
        shutil.rmtree(render_dir)

    img_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)

    for idx, image in enumerate(images):
        Image.fromarray(image).save(img_dir / f"{idx:02d}.jpeg")

    torch.save(torch.from_numpy(pix2face).long(), pix_path)
    torch.save(torch.from_numpy(point2face).long(), p2f_path)

    meta = {
        "backend": backend,
        "num_views": int(images.shape[0]),
        "image_size": int(images.shape[1]),
        "num_points": int(point2face.shape[0]),
        "num_faces": int(faces.shape[0]),
        "num_vertices": int(verts.shape[0]),
    }
    torch.save(meta, render_dir / "render_meta.pt")

    if save_mesh:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        o3d.io.write_triangle_mesh(str(render_dir / "mesh_recon.ply"), mesh)


def render_item(args: argparse.Namespace, item_id: str) -> None:
    uid = uid_from_item_id(item_id)
    print(f"[item] {item_id}")
    points, colors, normals = load_points_metadata(args.data_root, args.point_source_root, uid)
    points_norm, _, _ = normalize_xyz(points)

    verts, faces, vertex_colors = reconstruct_mesh(
        points=points_norm,
        colors=colors,
        normals=normals,
        poisson_depth=args.poisson_depth,
        density_quantile=args.density_quantile,
        target_faces=args.target_faces,
    )
    point2face = assign_points_to_faces(points_norm, verts, faces)

    backend = choose_backend(args)
    if backend == "pytorch3d":
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
        images, pix2face = render_with_pytorch3d(
            verts=verts,
            faces=faces,
            vertex_colors=vertex_colors,
            num_views=args.num_views,
            image_size=args.image_size,
            camera_dist=args.camera_dist,
            device=device,
        )
    else:
        images, pix2face = render_with_open3d(
            verts=verts,
            faces=faces,
            vertex_colors=vertex_colors,
            num_views=args.num_views,
            image_size=args.image_size,
            camera_dist=args.camera_dist,
            fov_deg=args.fov_deg,
        )

    save_outputs(
        data_root=args.data_root,
        item_id=item_id,
        uid=uid,
        images=images,
        pix2face=pix2face,
        point2face=point2face,
        verts=verts,
        faces=faces,
        overwrite=args.overwrite,
        save_mesh=args.save_mesh,
        backend=backend,
    )
    print(
        f"[done] {item_id} | backend={backend} | views={images.shape[0]} | "
        f"faces={faces.shape[0]} | verts={verts.shape[0]}"
    )


def main() -> None:
    args = parse_args()
    item_ids = select_item_ids(args)
    if not item_ids:
        raise ValueError("No item ids selected. Check --split/--start/--count or pass --item_ids.")

    for item_id in item_ids:
        render_item(args, item_id)


if __name__ == "__main__":
    main()
