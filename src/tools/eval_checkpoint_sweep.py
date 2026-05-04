#!/usr/bin/env python3
"""Batch evaluation for PatchAlign3D checkpoints.

Writes one JSON record per checkpoint and refreshes a compact CSV summary after
each successful or failed evaluation so long sweeps are resumable.
"""

import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

import open_clip

from bjfg.datasets.shapenet import PartNormalDataset
from bjfg.inference import eval as eval_lib
from bjfg.models.config import DEFAULT_BACKBONE_CONFIG

DEFAULT_OUTPUT = "logs/evals/checkpoint_eval_history.jsonl"


def parse_args():
    p = argparse.ArgumentParser("Batch-evaluate PatchAlign3D checkpoints")
    p.add_argument("--ckpt_glob", type=str, nargs="+", required=True, help="Checkpoint glob(s), e.g. logs/run/checkpoints/epoch_*.pt")
    p.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Unified JSONL history file. Future dataset evaluations can append to the same file.",
    )
    p.add_argument("--dataset", type=str, default="shapenetpart", choices=["shapenetpart", "faust", "partslip", "scanobjectnn"])
    p.add_argument("--shapenet_root", type=str, default="")
    p.add_argument("--faust_npz", type=str, nargs="*", default=[])
    p.add_argument("--faust_npoints", type=int, default=2048)
    p.add_argument("--scanobjectnn_root", type=str, default="")
    p.add_argument("--scanobjectnn_npoints", type=int, default=2048)
    p.add_argument("--min_epoch", type=int, default=None, help="Only include checkpoints with epoch >= this value")
    p.add_argument("--max_epoch", type=int, default=None, help="Only include checkpoints with epoch <= this value")
    p.add_argument("--include_last", action="store_true", default=False, help="Include last.pt if present in the glob")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--text_setting", type=str, default=None, choices=["part_only", "part_plus_cat", "ensemble"])
    p.add_argument("--assign", type=str, default="nearest", choices=["nearest", "membership"])
    p.add_argument("--clip_model", type=str, default=None)
    p.add_argument("--clip_pretrained", type=str, default=None)
    p.add_argument("--clip_tau", type=float, default=None, help="Force a fixed tau instead of reading checkpoint temp")
    p.add_argument("--use_color", action="store_true", default=False)
    p.add_argument("--use_normal", action="store_true", default=False)
    p.add_argument("--num_group", type=int, default=None)
    p.add_argument("--group_size", type=int, default=None)
    p.add_argument("--patch_encoder_type", type=str, default=None, choices=["pointnet", "hybrid"])
    p.add_argument("--patch_ms_scales", type=str, default=None)
    p.add_argument("--patch_edge_k", type=int, default=None)
    p.add_argument("--patch_refine_layers", type=int, default=None)
    p.add_argument("--patch_refine_k", type=int, default=None)
    refiner = p.add_mutually_exclusive_group()
    refiner.add_argument("--disable_patch_refiner", dest="disable_patch_refiner", action="store_true")
    refiner.add_argument("--enable_patch_refiner", dest="disable_patch_refiner", action="store_false")
    p.set_defaults(disable_patch_refiner=None)
    p.add_argument("--sort_by", type=str, default="point_miou")
    p.add_argument("--no_resume", action="store_true", default=False, help="Re-evaluate checkpoints even if already present in the output JSONL")
    p.add_argument("--stop_on_error", action="store_true", default=False)
    p.add_argument("--no_progress", action="store_true", default=True)
    return p.parse_args()


def checkpoint_epoch(path):
    match = re.search(r"epoch_(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    if path.stem == "last":
        return 10 ** 9
    return None


def collect_checkpoints(patterns, min_epoch=None, max_epoch=None, include_last=False):
    paths = {}
    for pattern in patterns:
        for item in glob.glob(pattern):
            path = Path(item).resolve()
            if not path.is_file() or path.suffix != ".pt":
                continue
            epoch = checkpoint_epoch(path)
            if epoch is None:
                continue
            if path.stem == "last" and not include_last:
                continue
            if epoch != 10 ** 9:
                if min_epoch is not None and epoch < int(min_epoch):
                    continue
                if max_epoch is not None and epoch > int(max_epoch):
                    continue
            paths[str(path)] = path
    return sorted(paths.values(), key=lambda p: (checkpoint_epoch(p), str(p)))


def _resolve_path(value):
    if not value:
        return ""
    return str(Path(value).resolve())


def _resolve_paths(values):
    return [_resolve_path(v) for v in (values or [])]


def dataset_ref_from_args(dataset, shapenet_root="", faust_npz=None, faust_npoints=2048, scanobjectnn_root="", scanobjectnn_npoints=2048):
    if dataset == "shapenetpart":
        return _resolve_path(shapenet_root)
    if dataset == "scanobjectnn":
        root = _resolve_path(scanobjectnn_root)
        return f"{dataset}:npoints={int(scanobjectnn_npoints)}::{root}" if root else f"{dataset}:npoints={int(scanobjectnn_npoints)}"
    faust_paths = sorted(_resolve_paths(faust_npz))
    if not faust_paths:
        return f"{dataset}:npoints={int(faust_npoints)}"
    return f"{dataset}:npoints={int(faust_npoints)}::{';'.join(faust_paths)}"


def build_eval_group(cli_args, effective_args, clip_model_name, clip_pretrained, text_setting):
    return {
        "dataset": str(cli_args.dataset),
        "dataset_ref": dataset_ref_from_args(
            cli_args.dataset,
            shapenet_root=cli_args.shapenet_root,
            faust_npz=cli_args.faust_npz,
            faust_npoints=cli_args.faust_npoints,
            scanobjectnn_root=cli_args.scanobjectnn_root,
            scanobjectnn_npoints=cli_args.scanobjectnn_npoints,
        ),
        "assign": str(cli_args.assign),
        "text_setting": str(text_setting),
        "clip_model": str(clip_model_name),
        "clip_pretrained": str(clip_pretrained),
        "clip_tau_override": None if cli_args.clip_tau is None else float(cli_args.clip_tau),
        "use_normal": bool(cli_args.use_normal),
        "use_color": bool(cli_args.use_color),
        "num_group": int(effective_args.num_group),
        "group_size": int(effective_args.group_size),
        "patch_encoder_type": str(effective_args.patch_encoder_type),
        "patch_ms_scales": str(effective_args.patch_ms_scales),
        "patch_edge_k": int(effective_args.patch_edge_k),
        "patch_refine_layers": int(effective_args.patch_refine_layers),
        "patch_refine_k": int(effective_args.patch_refine_k),
        "disable_patch_refiner": bool(effective_args.disable_patch_refiner),
    }


def eval_key_for_record(record):
    ckpt = record.get("ckpt", "")
    if ckpt:
        ckpt = str(Path(ckpt).resolve())
    group = record.get("eval_group")
    if not isinstance(group, dict):
        backbone = record.get("backbone") or {}
        group = {
            "dataset": record.get("dataset", ""),
            "dataset_ref": record.get("dataset_ref", ""),
            "assign": record.get("assign", ""),
            "text_setting": record.get("text_setting", ""),
            "clip_model": record.get("clip_model", ""),
            "clip_pretrained": record.get("clip_pretrained", ""),
            "clip_tau_override": record.get("clip_tau_override", None),
            "use_normal": bool(record.get("use_normal", False)),
            "use_color": bool(record.get("use_color", False)),
            "num_group": backbone.get("num_group"),
            "group_size": backbone.get("group_size"),
            "patch_encoder_type": backbone.get("patch_encoder_type"),
            "patch_ms_scales": str(backbone.get("patch_ms_scales", "")),
            "patch_edge_k": backbone.get("patch_edge_k"),
            "patch_refine_layers": backbone.get("patch_refine_layers"),
            "patch_refine_k": backbone.get("patch_refine_k"),
            "disable_patch_refiner": backbone.get("disable_patch_refiner"),
        }
    return json.dumps({"ckpt": ckpt, "eval_group": group}, sort_keys=True, ensure_ascii=True)


def load_existing_records(output_path):
    done = {}
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            eval_key = eval_key_for_record(record)
            if record.get("ckpt"):
                record["ckpt"] = str(Path(record["ckpt"]).resolve())
            record["eval_key"] = eval_key
            done[eval_key] = record
    return done


def append_record(output_path, record):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_summary_csv(output_path, records, sort_by="point_miou"):
    summary_path = output_path.with_suffix(".csv")
    rows = []
    for record in records:
        metrics = record.get("metrics") or {}
        rows.append(
            {
                "dataset": record.get("dataset", ""),
                "dataset_ref": record.get("dataset_ref", ""),
                "epoch": record.get("epoch", ""),
                "ckpt": record.get("ckpt", ""),
                "status": record.get("status", ""),
                "patch_acc": metrics.get("patch_acc", ""),
                "point_acc": metrics.get("point_acc", ""),
                "point_miou": metrics.get("point_miou", ""),
                "point_ciou": metrics.get("point_ciou", ""),
                "tau": record.get("tau", ""),
                "clip_model": record.get("clip_model", ""),
                "text_setting": record.get("text_setting", ""),
                "error": record.get("error", ""),
            }
        )

    def sort_key(row):
        value = row.get(sort_by, "")
        metric_group = (str(row.get("dataset", "")), str(row.get("dataset_ref", "")))
        if isinstance(value, (int, float)):
            return metric_group + (0, -float(value), str(row.get("epoch", "")))
        try:
            return metric_group + (0, -float(value), str(row.get("epoch", "")))
        except (TypeError, ValueError):
            return metric_group + (1, 0.0, str(row.get("epoch", "")))

    rows.sort(key=sort_key)
    fieldnames = [
        "dataset",
        "dataset_ref",
        "epoch",
        "ckpt",
        "status",
        "patch_acc",
        "point_acc",
        "point_miou",
        "point_ciou",
        "tau",
        "clip_model",
        "text_setting",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_args_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def coalesce(cli_value, ckpt_args, key, default):
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(key, default)


def build_eval_namespace(cli_args, ckpt_args):
    return SimpleNamespace(
        num_group=coalesce(cli_args.num_group, ckpt_args, "num_group", DEFAULT_BACKBONE_CONFIG["num_group"]),
        group_size=coalesce(cli_args.group_size, ckpt_args, "group_size", DEFAULT_BACKBONE_CONFIG["group_size"]),
        patch_encoder_type=coalesce(
            cli_args.patch_encoder_type,
            ckpt_args,
            "patch_encoder_type",
            DEFAULT_BACKBONE_CONFIG["patch_encoder_type"],
        ),
        patch_ms_scales=coalesce(
            cli_args.patch_ms_scales,
            ckpt_args,
            "patch_ms_scales",
            DEFAULT_BACKBONE_CONFIG["patch_ms_scales"],
        ),
        patch_edge_k=coalesce(cli_args.patch_edge_k, ckpt_args, "patch_edge_k", DEFAULT_BACKBONE_CONFIG["patch_edge_k"]),
        patch_refine_layers=coalesce(
            cli_args.patch_refine_layers,
            ckpt_args,
            "patch_refine_layers",
            DEFAULT_BACKBONE_CONFIG["patch_refine_layers"],
        ),
        patch_refine_k=coalesce(
            cli_args.patch_refine_k,
            ckpt_args,
            "patch_refine_k",
            DEFAULT_BACKBONE_CONFIG["patch_refine_k"],
        ),
        disable_patch_refiner=coalesce(
            cli_args.disable_patch_refiner,
            ckpt_args,
            "disable_patch_refiner",
            DEFAULT_BACKBONE_CONFIG["disable_patch_refiner"],
        ),
        use_color=bool(cli_args.use_color),
        use_normal=bool(cli_args.use_normal),
    )


def resolve_tau(override_tau, ckpt):
    if override_tau is not None:
        tau = float(override_tau)
        return tau, 1.0 / max(tau, 1e-6), "cli"
    temp_state = ckpt.get("temp", None)
    if isinstance(temp_state, dict):
        if "log_scale" in temp_state:
            scale = float(temp_state["log_scale"].exp().item())
            tau = 1.0 / max(scale, 1e-6)
            return tau, scale, "checkpoint.log_scale"
        if "scale" in temp_state:
            scale = float(temp_state["scale"].item())
            tau = 1.0 / max(scale, 1e-6)
            return tau, scale, "checkpoint.scale"
    tau = 0.07
    return tau, 1.0 / tau, "default"


def clip_settings(cli_args, ckpt_args):
    return (
        coalesce(cli_args.clip_model, ckpt_args, "clip_model", "ViT-bigG-14"),
        coalesce(cli_args.clip_pretrained, ckpt_args, "clip_pretrained", "laion2b_s39b_b160k"),
        coalesce(cli_args.text_setting, ckpt_args, "text_setting", "part_only"),
    )


def load_clip_bundle(device, clip_model_name, clip_pretrained, text_setting, dataset, shapenet_root, use_normal):
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    text_dim = int(getattr(clip_model, "text_projection", None).shape[1] if hasattr(clip_model, "text_projection") else 512)
    bundle = {
        "clip_model": clip_model,
        "tokenizer": tokenizer,
        "text_dim": text_dim,
        "text_setting": text_setting,
        "clip_model_name": clip_model_name,
        "clip_pretrained": clip_pretrained,
    }
    if dataset == "shapenetpart":
        dataset_tmp = PartNormalDataset(root=shapenet_root, split="test", normal_channel=use_normal)
        seg_classes = dataset_tmp.seg_classes
        id2cat = {v: k for k, v in dataset_tmp.classes.items()}
        text_feats_50 = eval_lib.encode_text_from_part_names(
            seg_classes,
            id2cat,
            device=device,
            setting=text_setting,
            clip_model=clip_model,
            tokenizer=tokenizer,
        )
        bundle["text_feats_50"] = text_feats_50
    return bundle


def evaluate_checkpoint(cli_args, ckpt_path, ckpt, effective_args, eval_group, device, clip_bundle):
    model = eval_lib.build_model(effective_args, device=device)
    proj_in_dim = 384
    if "proj" in ckpt and isinstance(ckpt["proj"], dict) and "proj.weight" in ckpt["proj"]:
        proj_in_dim = int(ckpt["proj"]["proj.weight"].shape[1])
    proj = eval_lib.PatchToTextProj(in_dim=proj_in_dim, out_dim=int(clip_bundle["text_dim"])).to(device)

    model.load_state_dict(ckpt["model"], strict=False)
    if "proj" in ckpt:
        proj.load_state_dict(ckpt["proj"], strict=False)

    tau, scale, tau_source = resolve_tau(cli_args.clip_tau, ckpt)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": cli_args.dataset,
        "dataset_ref": eval_group["dataset_ref"],
        "ckpt": str(Path(ckpt_path).resolve()),
        "epoch": checkpoint_epoch(Path(ckpt_path)),
        "status": "ok",
        "tau": tau,
        "scale": scale,
        "tau_source": tau_source,
        "clip_model": clip_bundle["clip_model_name"],
        "clip_pretrained": clip_bundle["clip_pretrained"],
        "text_setting": clip_bundle["text_setting"],
        "batch_size": cli_args.batch_size,
        "workers": cli_args.workers,
        "assign": cli_args.assign,
        "clip_tau_override": eval_group["clip_tau_override"],
        "use_normal": bool(cli_args.use_normal),
        "use_color": bool(cli_args.use_color),
        "eval_group": eval_group,
        "backbone": {
            "num_group": int(effective_args.num_group),
            "group_size": int(effective_args.group_size),
            "patch_encoder_type": str(effective_args.patch_encoder_type),
            "patch_ms_scales": str(effective_args.patch_ms_scales),
            "patch_edge_k": int(effective_args.patch_edge_k),
            "patch_refine_layers": int(effective_args.patch_refine_layers),
            "patch_refine_k": int(effective_args.patch_refine_k),
            "disable_patch_refiner": bool(effective_args.disable_patch_refiner),
        },
    }
    record["eval_key"] = eval_key_for_record(record)

    if cli_args.dataset == "shapenetpart":
        metrics = eval_lib.evaluate_shapenet(
            model,
            proj,
            clip_bundle["text_feats_50"],
            tau,
            device,
            cli_args.shapenet_root,
            batch_size=cli_args.batch_size,
            num_workers=cli_args.workers,
            assign_mode=cli_args.assign,
            progress=not cli_args.no_progress,
            use_normal=cli_args.use_normal,
        )
    elif cli_args.dataset == "scanobjectnn":
        metrics = eval_lib.evaluate_scanobjectnn(
            model,
            proj,
            clip_bundle["clip_model"],
            clip_bundle["tokenizer"],
            tau,
            device,
            cli_args.scanobjectnn_root,
            batch_size=cli_args.batch_size,
            num_workers=cli_args.workers,
            assign_mode=cli_args.assign,
            progress=not cli_args.no_progress,
            npoints=cli_args.scanobjectnn_npoints,
            text_setting=clip_bundle["text_setting"],
        )
    else:
        metrics = eval_lib.evaluate_faust(
            model,
            proj,
            clip_bundle["clip_model"],
            clip_bundle["tokenizer"],
            tau,
            device,
            cli_args.faust_npz,
            batch_size=cli_args.batch_size,
            num_workers=cli_args.workers,
            assign_mode=cli_args.assign,
            progress=not cli_args.no_progress,
            npoints=cli_args.faust_npoints,
            text_setting=clip_bundle["text_setting"],
            dataset_name=cli_args.dataset,
        )
    record["metrics"] = metrics
    return record


def print_result(record):
    metrics = record.get("metrics", {})
    main_bits = [
        f"epoch={record.get('epoch')}",
        f"status={record.get('status')}",
        f"patch_acc={metrics.get('patch_acc', 'n/a')}",
        f"point_acc={metrics.get('point_acc', 'n/a')}",
        f"point_miou={metrics.get('point_miou', 'n/a')}",
    ]
    if "point_ciou" in metrics:
        main_bits.append(f"point_ciou={metrics.get('point_ciou')}")
    print(" | ".join(main_bits), flush=True)


def main():
    args = parse_args()
    if args.dataset == "shapenetpart" and not args.shapenet_root:
        raise ValueError("--shapenet_root is required for ShapeNetPart evaluation")
    if args.dataset == "scanobjectnn" and not args.scanobjectnn_root:
        raise ValueError("--scanobjectnn_root is required for ScanObjectNN evaluation")
    if args.dataset in ("faust", "partslip") and not args.faust_npz:
        raise ValueError("--faust_npz is required for FAUST/PartSLIP evaluation")

    checkpoints = collect_checkpoints(
        args.ckpt_glob,
        min_epoch=args.min_epoch,
        max_epoch=args.max_epoch,
        include_last=args.include_last,
    )
    if not checkpoints:
        raise RuntimeError("No checkpoints matched the provided patterns")

    output_path = Path(args.output).resolve()
    existing = {} if args.no_resume else load_existing_records(output_path)
    if existing:
        print(f"[resume] loaded {len(existing)} existing records from {output_path}", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    clip_cache = {}
    all_records = list(existing.values())
    planned_eval_keys = set()

    for ckpt_path in checkpoints:
        ckpt_key = str(Path(ckpt_path).resolve())

        ckpt_preview = torch.load(ckpt_path, map_location="cpu")
        ckpt_args = to_args_dict(ckpt_preview.get("args"))
        effective_args = build_eval_namespace(args, ckpt_args)
        clip_model_name, clip_pretrained, text_setting = clip_settings(args, ckpt_args)
        eval_group = build_eval_group(args, effective_args, clip_model_name, clip_pretrained, text_setting)
        eval_key = json.dumps({"ckpt": ckpt_key, "eval_group": eval_group}, sort_keys=True, ensure_ascii=True)
        planned_eval_keys.add(eval_key)
        if eval_key in existing:
            print(f"[skip] already recorded: {ckpt_key}", flush=True)
            continue

        clip_key = (
            args.dataset,
            eval_group["dataset_ref"],
            clip_model_name,
            clip_pretrained,
            text_setting,
            bool(args.use_normal),
        )
        if clip_key not in clip_cache:
            print(
                f"[clip] loading model={clip_model_name} pretrained={clip_pretrained} text_setting={text_setting}",
                flush=True,
            )
            clip_cache[clip_key] = load_clip_bundle(
                device=device,
                clip_model_name=clip_model_name,
                clip_pretrained=clip_pretrained,
                text_setting=text_setting,
                dataset=args.dataset,
                shapenet_root=args.shapenet_root,
                use_normal=args.use_normal,
            )

        print(f"[eval] {ckpt_key}", flush=True)
        try:
            record = evaluate_checkpoint(args, ckpt_path, ckpt_preview, effective_args, eval_group, device, clip_cache[clip_key])
        except Exception as exc:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": args.dataset,
                "dataset_ref": eval_group["dataset_ref"],
                "ckpt": ckpt_key,
                "epoch": checkpoint_epoch(Path(ckpt_path)),
                "status": "error",
                "assign": args.assign,
                "text_setting": text_setting,
                "clip_model": clip_model_name,
                "clip_pretrained": clip_pretrained,
                "clip_tau_override": eval_group["clip_tau_override"],
                "use_normal": bool(args.use_normal),
                "use_color": bool(args.use_color),
                "eval_group": eval_group,
                "error": f"{type(exc).__name__}: {exc}",
            }
            record["eval_key"] = eval_key_for_record(record)
            print(f"[error] {ckpt_key}: {record['error']}", flush=True)
            append_record(output_path, record)
            all_records.append(record)
            write_summary_csv(output_path, all_records, sort_by=args.sort_by)
            if args.stop_on_error:
                raise
            continue

        append_record(output_path, record)
        all_records.append(record)
        write_summary_csv(output_path, all_records, sort_by=args.sort_by)
        print_result(record)
        torch.cuda.empty_cache()

    ok_records = [r for r in all_records if r.get("status") == "ok" and eval_key_for_record(r) in planned_eval_keys]
    if ok_records:
        best = max(ok_records, key=lambda r: float((r.get("metrics") or {}).get(args.sort_by, float("-inf"))))
        best_metric = (best.get("metrics") or {}).get(args.sort_by)
        print(f"[best] sort_by={args.sort_by} epoch={best.get('epoch')} value={best_metric}", flush=True)
        print(f"[best] ckpt={best.get('ckpt')}", flush=True)
    print(f"[done] wrote records to {output_path}", flush=True)
    print(f"[done] wrote summary to {output_path.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
