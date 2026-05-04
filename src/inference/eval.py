

import argparse
import json
import os
import re
from pathlib import Path

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bjfg.datasets.shapenet import PartNormalDataset
from bjfg.models.config import add_backbone_args, build_backbone_config
from bjfg.models import point_transformer

import open_clip


# Canonical part-name candidates (global ShapeNetPart ids)
PART_NAME_CANDIDATES = {
    "Airplane": ["body", "wing", "tail", "engine or frame"],
    "Chair": ["back", "seat", "leg", "arm"],
    "Car": ["roof", "hood", "wheel", "body"],
    "Table": ["desktop", "leg or support", "drawer"],
    "Lamp": ["base", "lampshade", "fixing bracket", "pole"],
    "Guitar": ["headstock", "neck", "body"],
    "Rocket": ["body", "fin", "nose"],
    "Pistol": ["barrel", "handle or grip", "trigger and guard"],
    "Skateboard": ["wheel", "deck", "belt for foot"],
    "Bag": ["handle", "body"],
    "Cap": ["crown", "brim"],
    "Laptop": ["keyboard", "screen"],
    "Mug": ["handle", "cup"],
    "Knife": ["blade", "handle"],
    "Earphone": ["earcup", "headband", "data wire"],
    "Motorbike": ["gas tank", "seat", "wheel", "handles or handlebars", "headlight", "engine or frame"],
}

PART_ONLY_TEMPLATES = [
    "{}",
    "a {}",
    "{} part",
]

PART_PLUS_CAT_TEMPLATES = [
    "a {} of a {}",
    "the {} of a {}",
    "{} of {}",
    "a {} part of a {}",
]

SCANOBJECTNN_ID2CATEGORY = {
    0: "bag",
    1: "bin",
    2: "box",
    3: "cabinet",
    4: "chair",
    5: "desk",
    6: "display",
    7: "door",
    8: "shelf",
    9: "table",
    10: "bed",
    11: "pillow",
    12: "sink",
    13: "sofa",
    14: "toilet",
}

SCANOBJECTNN_PART_ID_REMAP = {
    0: torch.arange(4),
    1: torch.tensor([0, 1, 2, -1, 3]),
    2: torch.arange(5),
    3: torch.arange(7),
    4: torch.tensor([0, -1, 1, 2, 3, 4]),
    5: torch.arange(4),
    6: torch.arange(3),
    7: torch.arange(3),
    8: torch.arange(4),
    9: torch.arange(3),
    10: torch.arange(3),
    11: torch.arange(2),
    12: torch.arange(4),
    13: torch.arange(5),
    14: torch.arange(6),
}


def _clean_text(s):
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pc_normalize_np(pc):
    pc = np.asarray(pc, dtype=np.float32)
    if pc.size == 0:
        return pc
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc = pc - centroid
    scale = float(np.max(np.sqrt(np.sum(pc ** 2, axis=1))))
    if scale > 0:
        pc = pc / scale
    return pc.astype(np.float32, copy=False)


def _deterministic_choice(num_points, npoints, seed):
    if npoints <= 0 or num_points == npoints:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    replace = num_points < npoints
    return np.asarray(rng.choice(num_points, size=npoints, replace=replace), dtype=np.int64)


def _load_scanobjectnn_part_names():
    prompt_path = Path(__file__).resolve().parents[2] / "cops" / "source" / "prompts" / "ScanObjectNN-Part_part_names.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"ScanObjectNN prompt file not found: {prompt_path}")
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    return {str(cat): [str(x) for x in names] for cat, names in data.items()}


class ScanObjectNNPartDataset(Dataset):
    def __init__(self, root, npoints=2048, split="test", normalize=True, seed=42):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}")
        base = Path(root).resolve()
        if (base / "object_dataset_complete_with_parts_").is_dir():
            base = base / "object_dataset_complete_with_parts_"
        if not (base / "split_new.txt").exists():
            raise FileNotFoundError(f"ScanObjectNN split file not found under: {base}")
        self.base = base
        self.npoints = int(npoints)
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.part_names = _load_scanobjectnn_part_names()
        samples = []
        split_path = self.base / "split_new.txt"
        for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            file_name = cols[0].strip()
            category_id = int(cols[1])
            is_test = len(cols) >= 3 and cols[2].strip() == "t"
            if split == "test" and not is_test:
                continue
            if split == "train" and is_test:
                continue
            samples.append((file_name, category_id))
        if not samples:
            raise RuntimeError(f"No ScanObjectNN samples found for split={split} under {self.base}")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, category_id = self.samples[idx]
        category_name = SCANOBJECTNN_ID2CATEGORY[int(category_id)]
        stem = file_name[:-4] if file_name.endswith(".bin") else file_name
        points_raw = np.fromfile(self.base / category_name / f"{stem}.bin", dtype=np.float32)
        labels_raw = np.fromfile(self.base / category_name / f"{stem}_part.bin", dtype=np.float32)
        points = points_raw[1:].reshape((-1, 11))[:, :3].astype(np.float32)
        labels = labels_raw[1:].reshape((-1, 2))[:, -1].astype(np.int64)
        remap = SCANOBJECTNN_PART_ID_REMAP[int(category_id)].cpu().numpy()
        mapped = np.full(labels.shape, -1, dtype=np.int64)
        valid = (labels >= 0) & (labels < len(remap))
        mapped[valid] = remap[labels[valid]]
        if self.npoints > 0 and points.shape[0] != self.npoints:
            sel = _deterministic_choice(points.shape[0], self.npoints, self.seed + idx)
            points = points[sel]
            mapped = mapped[sel]
        if self.normalize:
            points = _pc_normalize_np(points)
        return {
            "points": points,
            "labels": mapped,
            "label_names": self.part_names[category_name],
            "category": category_name,
            "slug": stem,
        }


class PatchToTextProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb):
        x = patch_emb.transpose(1, 2)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


def build_model(args, device):
    cfg = build_backbone_config(args, color=bool(args.use_color or args.use_normal))
    model = point_transformer.get_model(cfg).to(device)
    return model


def prepare_points(points):
    if points.ndim != 3:
        raise ValueError(f"Expected (B,N,C) points, got shape {points.shape}")
    pts = points.transpose(2, 1).contiguous()
    pts[:, [1, 2], :] = pts[:, [2, 1], :]
    return pts


def compute_patch_targets_vector(point_labels, patch_idx, num_labels):
    G, M = patch_idx.shape
    gathered = point_labels.gather(0, patch_idx.reshape(-1)).view(G, M)
    lbl = (gathered + 1).clamp_min(0)
    K = int(num_labels) + 1
    one_hot = F.one_hot(lbl.clamp_max(K - 1), num_classes=K).sum(dim=1)
    one_hot[:, 0] = 0
    has_any = one_hot.sum(dim=1) > 0
    preds = one_hot.argmax(dim=1) - 1
    out = torch.full((G,), -1, dtype=torch.long, device=point_labels.device)
    out[has_any] = preds[has_any]
    return out


def assign_points_from_patches(points_xyz, patch_centers, patch_logits, patch_idx, mode="nearest"):
    B, _, N = points_xyz.shape
    K = patch_logits.shape[-1]
    if mode == "membership":
        point_logits = torch.zeros(B, N, K, device=points_xyz.device)
        counts = torch.zeros(B, N, 1, device=points_xyz.device)
        for b in range(B):
            idx = patch_idx[b].reshape(-1)
            src = patch_logits[b].unsqueeze(1).expand_as(patch_idx[b].unsqueeze(-1)).reshape(-1, K)
            point_logits[b].index_add_(0, idx, src)
            ones = torch.ones(idx.shape[0], 1, device=points_xyz.device)
            counts[b].index_add_(0, idx, ones)
        return point_logits / counts.clamp_min(1.0)
    from knn_cuda import KNN
    knn = KNN(k=1, transpose_mode=True)
    _, nearest = knn(patch_centers.transpose(1, 2).contiguous(), points_xyz.transpose(1, 2).contiguous())
    nearest = nearest.squeeze(-1)
    return patch_logits.gather(1, nearest.unsqueeze(-1).expand(-1, -1, K))


def compute_point_metrics(point_pred, target, label, seg_classes, id2cat):
    """Returns accuracy, per-instance IoUs, and per-category IoUs (instance-averaged).

    Note: If a part is absent in GT for that sample, it is skipped (no IoU added),
    to avoid inflating IoU when union is zero.
    """
    B, N = target.shape
    acc = (point_pred == target).float().mean().item()
    inst_ious = []
    cat_to_ious = {cat: [] for cat in seg_classes}
    for b in range(B):
        cat = id2cat[int(label[b].item())]
        valid = seg_classes[cat]
        preds = point_pred[b]
        gts = target[b]
        ious = []
        for pid in valid:
            pred_mask = preds == pid
            gt_mask = gts == pid
            if gt_mask.sum().item() == 0:
                # part not present in GT -> skip to avoid over-estimating IoU
                continue
            inter = (pred_mask & gt_mask).sum().item()
            union = (pred_mask | gt_mask).sum().item()
            if union == 0:
                continue
            iou = inter / union
            ious.append(iou)
        if ious:
            inst = sum(ious) / len(ious)
            inst_ious.append(inst)
            cat_to_ious[cat].append(inst)
    return dict(
        acc=acc,
        inst_ious=inst_ious,
        cat_to_ious=cat_to_ious,
        per_cat_iou={k: (sum(v) / len(v)) for k, v in cat_to_ious.items() if v},
    )


def compute_point_metrics_generic(point_pred, target, num_labels, average_mode="all_labels"):
    """Instance-weighted IoU for datasets without category structure.

    `average_mode="all_labels"` is the generic behavior used for FAUST-like
    datasets. `average_mode="gt_present"` mirrors the PartSLIP-style shape
    mIoU: average IoU only over parts that appear in the ground truth for that
    shape, while ignoring unlabeled points (`target < 0`).
    """
    B, N = target.shape
    valid = target >= 0
    valid_count = valid.sum().item()
    acc = (((point_pred == target) & valid).float().sum().item() / valid_count) if valid_count > 0 else 0.0
    inst_ious = []
    part_to_ious = {i: [] for i in range(num_labels)}
    for b in range(B):
        preds = point_pred[b]
        gts = target[b]
        valid_b = gts >= 0
        ious = []
        for pid in range(num_labels):
            pred_mask = (preds == pid) & valid_b
            gt_mask = (gts == pid) & valid_b
            if average_mode == "gt_present" and gt_mask.sum().item() == 0:
                continue
            inter = (pred_mask & gt_mask).sum().item()
            union = (pred_mask | gt_mask).sum().item()
            if union == 0:
                if average_mode == "gt_present":
                    continue
                iou = 1.0
            else:
                iou = inter / union
            ious.append(iou)
            part_to_ious[pid].append(iou)
        if ious:
            inst_ious.append(sum(ious) / len(ious))
    return dict(
        acc=acc,
        inst_ious=inst_ious,
        part_to_ious={k: (sum(v) / len(v)) for k, v in part_to_ious.items() if v},
    )


@torch.no_grad()
def encode_texts(names, category, setting, clip_model, tokenizer, device):
    texts = []
    cname = _clean_text(category or "")
    for nm in names:
        nm = _clean_text(nm)
        if setting in ("part_plus_cat", "ensemble") and cname:
            for tpl in PART_PLUS_CAT_TEMPLATES:
                slots = tpl.count("{}")
                if slots == 2:
                    texts.append(tpl.format(nm, cname))
                elif slots == 1:
                    texts.append(tpl.format(f"{cname} {nm}"))
                else:
                    texts.append(f"{cname} {nm}")
        if (setting in ("part_only", "ensemble")) or not cname:
            for tpl in PART_ONLY_TEMPLATES:
                texts.append(tpl.format(nm) if tpl.count("{}") == 1 else nm)
    if not texts:
        return torch.zeros(clip_model.text_projection.shape[1], device=device)
    toks = tokenizer(texts).to(device)
    feat = clip_model.encode_text(toks)
    feat = F.normalize(feat, dim=-1)
    return F.normalize(feat.mean(dim=0, keepdim=True), dim=-1).squeeze(0)


@torch.no_grad()
def encode_text_from_part_names(seg_classes, id2cat, device, setting, clip_model, tokenizer):
    bank = torch.zeros(50, clip_model.text_projection.shape[1], device=device)
    filled = torch.zeros(50, dtype=torch.bool, device=device)
    for cid, cat in id2cat.items():
        if cat not in seg_classes:
            continue
        gids = list(sorted(seg_classes[cat]))
        cand = PART_NAME_CANDIDATES.get(cat, [])
        for idx, gid in enumerate(gids):
            pname = cand[idx] if idx < len(cand) else f"part{idx}"
            bank[gid] = encode_texts([pname], category=cat if setting != "part_only" else None, setting=setting, clip_model=clip_model, tokenizer=tokenizer, device=device)
            filled[gid] = True
    if (~filled).any():
        mean_vec = F.normalize(bank[filled].mean(dim=0, keepdim=True), dim=-1) if filled.any() else F.normalize(torch.randn(1, bank.shape[1], device=device), dim=-1)
        bank = torch.where(filled.unsqueeze(-1), bank, mean_vec.expand_as(bank))
    return bank


def evaluate_shapenet(model, proj, text_feats_50, tau, device, shapenet_root, batch_size, num_workers, assign_mode, progress, use_normal):
    dataset = PartNormalDataset(root=shapenet_root, npoints=2048, split="test", normal_channel=use_normal)
    seg_classes = dataset.seg_classes
    id2cat = {v: k for k, v in dataset.classes.items()}
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    model.eval()
    proj.eval()
    tot_patch = tot_patch_correct = 0
    pt_acc = 0.0
    inst_iou_sum = 0.0
    inst_count = 0
    cat_to_ious = {cat: [] for cat in seg_classes}
    iterator = tqdm(dl, total=len(dl), desc="ShapeNetPart", smoothing=0.9) if progress else dl
    with torch.no_grad():
        for points, label, target in iterator:
            points = points.to(device).float()
            label = label.to(device).long()
            target = target.to(device).long()
            points = prepare_points(points)
            xyz = points[:, :3, :]
            pe, pc, pi = model.forward_patches(points)
            x = proj(pe)
            logits = (x @ text_feats_50.t()) / max(tau, 1e-6)
            valid = torch.zeros_like(logits, dtype=torch.bool)
            for b in range(points.size(0)):
                cat = id2cat[int(label[b].item())]
                allow = seg_classes.get(cat, [])
                if allow:
                    valid[b, :, allow] = True
            logits = logits.masked_fill(~valid, -1e4)
            patch_pred = logits.argmax(dim=-1)
            for b in range(points.size(0)):
                patch_gt = compute_patch_targets_vector(target[b], pi[b], 50)
                present = patch_gt >= 0
                tot_patch += present.sum().item()
                tot_patch_correct += (patch_pred[b][present] == patch_gt[present]).sum().item()
            point_logits = assign_points_from_patches(xyz, pc, logits, pi, mode=assign_mode)
            pred = point_logits.argmax(dim=-1)
            metrics_batch = compute_point_metrics(pred, target, label, seg_classes, id2cat)
            pt_acc += metrics_batch["acc"]
            inst_iou_sum += sum(metrics_batch["inst_ious"])
            inst_count += len(metrics_batch["inst_ious"])
            for cat, v in metrics_batch["per_cat_iou"].items():
                cat_to_ious[cat].append(v)
    num_batches = len(dl)
    per_cat_iou = {k: sum(v) / len(v) for k, v in cat_to_ious.items() if v}
    ciou_vals = list(per_cat_iou.values())
    metrics = {
        "patch_acc": tot_patch_correct / max(tot_patch, 1),
        "point_acc": pt_acc / max(num_batches, 1),
        "point_miou": inst_iou_sum / max(inst_count, 1),
        "point_ciou": (sum(ciou_vals) / len(ciou_vals)) if ciou_vals else 0.0,
        "per_cat_iou": per_cat_iou,
    }
    return metrics


class FaustNpzDataset(torch.utils.data.Dataset):
    """Loads FAUST-style files with keys: points (N,3), labels (N,), label_names (K)."""

    def __init__(self, paths, npoints=2048):
        files = []
        for p in paths:
            cand = []
            if any(ch in p for ch in "*?[]"):
                cand = glob.glob(p)
            else:
                cand = [p]
            for c in cand:
                path = Path(c)
                if path.is_dir():
                    files.extend(sorted(path.glob("*.npz")))
                elif path.suffix == ".npz" and path.exists():
                    files.append(path)
        if not files:
            raise RuntimeError("No NPZ files found for FAUST evaluation.")
        self.files = files
        self.npoints = int(npoints)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        pts = np.asarray(data["points"], dtype=np.float32)
        labels = np.asarray(data["labels"]).reshape(-1).astype(np.int64)
        names = [str(x) for x in data.get("label_names", [])]
        category = data.get("category", "")
        if isinstance(category, np.ndarray):
            category = category.item() if category.shape == () else category.tolist()
        category = str(category) if category is not None else ""
        if names and labels.max(initial=-1) >= len(names):
            labels = np.clip(labels, 0, len(names) - 1)
        if self.npoints > 0 and pts.shape[0] != self.npoints:
            N = pts.shape[0]
            replace = N < self.npoints
            sel = np.random.choice(N, size=self.npoints, replace=replace)
            pts = pts[sel]
            labels = labels[sel]
        return {"points": pts[:, :3], "labels": labels, "label_names": names, "category": category, "slug": path.stem}


def collate_faust(batch):
    return batch


def evaluate_faust(
    model,
    proj,
    clip_model,
    tokenizer,
    tau,
    device,
    npz_paths,
    batch_size,
    num_workers,
    assign_mode,
    progress,
    npoints,
    text_setting="part_only",
    dataset_name="faust",
):
    dataset = FaustNpzDataset(npz_paths, npoints=npoints)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collate_faust)
    model.eval()
    proj.eval()
    tot_patch = tot_patch_correct = 0
    acc_sum = 0.0
    sample_count = 0
    inst_iou_sum = 0.0
    inst_count = 0
    part_to_ious = {}
    iterator = tqdm(dl, total=len(dl), desc="FAUST", smoothing=0.9) if progress else dl
    with torch.no_grad():
        for batch in iterator:
            points_list = []
            labels_list = []
            names_list = []
            category_list = []
            for sample in batch:
                pts = torch.as_tensor(sample["points"], dtype=torch.float32, device=device).unsqueeze(0)  # (1,N,3)
                labels = torch.as_tensor(sample["labels"], dtype=torch.long, device=device)
                points_list.append(pts)
                labels_list.append(labels)
                names_list.append(sample.get("label_names", []))
                category_list.append(str(sample.get("category", "") or ""))
            points = torch.cat(points_list, dim=0)  # (B,N,3)
            labels = labels_list
            B, N, _ = points.shape
            points_pt = prepare_points(points)  # (B,3,N)
            pe, pc, pi = model.forward_patches(points_pt)
            x = proj(pe)
            for b in range(B):
                names = names_list[b]
                if not names:
                    continue
                category = category_list[b] if b < len(category_list) else ""
                # Encode each part name separately to get (K,D)
                text_feats = torch.stack(
                    [
                        encode_texts(
                            [nm],
                            category=category if category else None,
                            setting=text_setting,
                            clip_model=clip_model,
                            tokenizer=tokenizer,
                            device=device,
                        )
                        for nm in names
                    ],
                    dim=0,
                )
                logits = (x[b] @ text_feats.t()) / max(tau, 1e-6)  # (G,K)
                patch_pred = logits.argmax(dim=-1)
                patch_gt = compute_patch_targets_vector(labels[b], pi[b], len(names))
                present = patch_gt >= 0
                tot_patch += present.sum().item()
                tot_patch_correct += (patch_pred[present] == patch_gt[present]).sum().item()
                # Point assignment
                point_logits = assign_points_from_patches(points_pt[b : b + 1, :3, :], pc[b : b + 1], logits.unsqueeze(0), pi[b : b + 1], mode=assign_mode)
                pred = point_logits.argmax(dim=-1).squeeze(0)
                metrics_b = compute_point_metrics_generic(
                    pred.unsqueeze(0),
                    labels[b].unsqueeze(0),
                    len(names),
                    average_mode="gt_present" if dataset_name == "partslip" else "all_labels",
                )
                acc_sum += metrics_b["acc"]
                sample_count += 1
                inst_iou_sum += sum(metrics_b["inst_ious"])
                inst_count += len(metrics_b["inst_ious"])
                for pid, v in metrics_b["part_to_ious"].items():
                    part_to_ious.setdefault(pid, []).append(v)
    per_part_iou = {k: sum(v) / len(v) for k, v in part_to_ious.items() if v}
    metrics = {
        "patch_acc": tot_patch_correct / max(tot_patch, 1),
        "point_acc": acc_sum / max(sample_count, 1),
        "point_miou": inst_iou_sum / max(inst_count, 1),
        "per_part_iou": per_part_iou,
    }
    return metrics


def evaluate_scanobjectnn(
    model,
    proj,
    clip_model,
    tokenizer,
    tau,
    device,
    scanobjectnn_root,
    batch_size,
    num_workers,
    assign_mode,
    progress,
    npoints,
    text_setting="part_only",
):
    dataset = ScanObjectNNPartDataset(scanobjectnn_root, npoints=npoints, split="test", normalize=True)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collate_faust)
    model.eval()
    proj.eval()
    tot_patch = tot_patch_correct = 0
    acc_sum = 0.0
    sample_count = 0
    inst_iou_sum = 0.0
    inst_count = 0
    cat_to_ious = {cat: [] for cat in dataset.part_names}
    iterator = tqdm(dl, total=len(dl), desc="ScanObjectNN", smoothing=0.9) if progress else dl
    with torch.no_grad():
        for batch in iterator:
            points_list = []
            labels_list = []
            names_list = []
            category_list = []
            for sample in batch:
                pts = torch.as_tensor(sample["points"], dtype=torch.float32, device=device).unsqueeze(0)
                labels = torch.as_tensor(sample["labels"], dtype=torch.long, device=device)
                points_list.append(pts)
                labels_list.append(labels)
                names_list.append(sample.get("label_names", []))
                category_list.append(str(sample.get("category", "") or ""))
            points = torch.cat(points_list, dim=0)
            labels = labels_list
            points_pt = prepare_points(points)
            pe, pc, pi = model.forward_patches(points_pt)
            x = proj(pe)
            for b in range(points.shape[0]):
                names = names_list[b]
                category = category_list[b] if b < len(category_list) else ""
                if not names or not category:
                    continue
                text_feats = torch.stack(
                    [
                        encode_texts(
                            [nm],
                            category=category if category else None,
                            setting=text_setting,
                            clip_model=clip_model,
                            tokenizer=tokenizer,
                            device=device,
                        )
                        for nm in names
                    ],
                    dim=0,
                )
                logits = (x[b] @ text_feats.t()) / max(tau, 1e-6)
                patch_pred = logits.argmax(dim=-1)
                patch_gt = compute_patch_targets_vector(labels[b], pi[b], len(names))
                present = patch_gt >= 0
                tot_patch += present.sum().item()
                tot_patch_correct += (patch_pred[present] == patch_gt[present]).sum().item()
                point_logits = assign_points_from_patches(points_pt[b : b + 1, :3, :], pc[b : b + 1], logits.unsqueeze(0), pi[b : b + 1], mode=assign_mode)
                pred = point_logits.argmax(dim=-1).squeeze(0)
                metrics_b = compute_point_metrics_generic(
                    pred.unsqueeze(0),
                    labels[b].unsqueeze(0),
                    len(names),
                    average_mode="gt_present",
                )
                acc_sum += metrics_b["acc"]
                sample_count += 1
                if metrics_b["inst_ious"]:
                    inst_iou = float(metrics_b["inst_ious"][0])
                    inst_iou_sum += inst_iou
                    inst_count += 1
                    cat_to_ious.setdefault(category, []).append(inst_iou)
    per_cat_iou = {k: sum(v) / len(v) for k, v in cat_to_ious.items() if v}
    metrics = {
        "patch_acc": tot_patch_correct / max(tot_patch, 1),
        "point_acc": acc_sum / max(sample_count, 1),
        "point_miou": inst_iou_sum / max(inst_count, 1),
        "point_ciou": sum(per_cat_iou.values()) / max(len(per_cat_iou), 1),
        "per_cat_iou": per_cat_iou,
    }
    return metrics


def parse_args():
    p = argparse.ArgumentParser("Evaluate PatchAlign3D checkpoint (ShapeNetPart, FAUST)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--shapenet_root", type=str, required=False, default="", help="Path to ShapeNetPart root")
    p.add_argument("--faust_npz", type=str, nargs="*", default=[], help="FAUST NPZ file(s) or directory(ies)")
    p.add_argument("--faust_npoints", type=int, default=2048)
    p.add_argument("--scanobjectnn_root", type=str, default="", help="Path to ScanObjectNN root")
    p.add_argument("--scanobjectnn_npoints", type=int, default=2048)
    p.add_argument("--dataset_name", type=str, default="faust", choices=["faust", "partslip"])
    p.add_argument("--text_setting", type=str, default="part_only", choices=["part_only", "part_plus_cat", "ensemble"])
    p.add_argument("--assign", type=str, default="nearest", choices=["nearest", "membership"])
    p.add_argument("--clip_model", type=str, default="ViT-bigG-14")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s39b_b160k")
    p.add_argument("--clip_tau", type=float, default=None, help="Overrides tau. If omitted, try to load learned tau from checkpoint temp.")
    p.add_argument("--batch_size", type=int, default=16)
    add_backbone_args(p)
    p.add_argument("--use_color", action="store_true", default=False)
    p.add_argument("--use_normal", action="store_true", default=False)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_progress", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text_dim = int(getattr(clip_model, "text_projection", None).shape[1] if hasattr(clip_model, "text_projection") else 512)

    model = build_model(args, device=device)
    proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        res = model.load_state_dict(ckpt["model"], strict=False)
        missing = getattr(res, "missing_keys", res[0] if isinstance(res, (list, tuple)) else [])
        unexpected = getattr(res, "unexpected_keys", res[1] if isinstance(res, (list, tuple)) else [])
        print(f"[ckpt] model loaded: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing)
        if unexpected:
            print("  unexpected:", unexpected)
    if "proj" in ckpt:
        res = proj.load_state_dict(ckpt["proj"], strict=False)
        missing = getattr(res, "missing_keys", res[0] if isinstance(res, (list, tuple)) else [])
        unexpected = getattr(res, "unexpected_keys", res[1] if isinstance(res, (list, tuple)) else [])
        print(f"[ckpt] proj loaded: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing)
        if unexpected:
            print("  unexpected:", unexpected)

    tau = None
    if args.clip_tau is not None:
        tau = float(args.clip_tau)
        print(f"[tau] using CLI override: {tau:.6f}")
    else:
        temp_state = ckpt.get("temp", None)
        if isinstance(temp_state, dict):
            if "log_scale" in temp_state:
                scale = float(temp_state["log_scale"].exp().item())
                tau = 1.0 / max(scale, 1e-6)
                print(f"[tau] loaded from checkpoint temp.log_scale: scale={scale:.6f} tau={tau:.6f}")
            elif "scale" in temp_state:
                scale = float(temp_state["scale"].item())
                tau = 1.0 / max(scale, 1e-6)
                print(f"[tau] loaded from checkpoint temp.scale: scale={scale:.6f} tau={tau:.6f}")
    if tau is None:
        tau = 0.07
        print(f"[tau] checkpoint temp missing, fallback to default: {tau:.6f}")

    if args.shapenet_root:
        dataset_tmp = PartNormalDataset(root=args.shapenet_root, split="test", normal_channel=args.use_normal)
        seg_classes = dataset_tmp.seg_classes
        id2cat = {v: k for k, v in dataset_tmp.classes.items()}
        text_feats_50 = encode_text_from_part_names(seg_classes, id2cat, device=device, setting=args.text_setting, clip_model=clip_model, tokenizer=tokenizer)

        metrics_sh = evaluate_shapenet(
            model,
            proj,
            text_feats_50,
            tau,
            device,
            args.shapenet_root,
            batch_size=args.batch_size,
            num_workers=args.workers,
            assign_mode=args.assign,
            progress=not args.no_progress,
            use_normal=args.use_normal,
        )
        print("ShapeNetPart metrics:", metrics_sh)

    if args.faust_npz:
        metrics_faust = evaluate_faust(
            model,
            proj,
            clip_model,
            tokenizer,
            tau,
            device,
            args.faust_npz,
            batch_size=args.batch_size,
            num_workers=args.workers,
            assign_mode=args.assign,
            progress=not args.no_progress,
            npoints=args.faust_npoints,
            text_setting=args.text_setting,
            dataset_name=args.dataset_name,
        )
        print("FAUST metrics:", metrics_faust)

    if args.scanobjectnn_root:
        metrics_scan = evaluate_scanobjectnn(
            model,
            proj,
            clip_model,
            tokenizer,
            tau,
            device,
            args.scanobjectnn_root,
            batch_size=args.batch_size,
            num_workers=args.workers,
            assign_mode=args.assign,
            progress=not args.no_progress,
            npoints=args.scanobjectnn_npoints,
            text_setting=args.text_setting,
        )
        print("ScanObjectNN metrics:", metrics_scan)


if __name__ == "__main__":
    main()
