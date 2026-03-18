from easydict import EasyDict


DEFAULT_BACKBONE_CONFIG = {
    "trans_dim": 384,
    "depth": 12,
    "drop_path_rate": 0.1,
    "cls_dim": 50,
    "num_heads": 6,
    "group_size": 32,
    "num_group": 128,
    "encoder_dims": 256,
    "color": False,
    "num_classes": 16,
    "patch_encoder_type": "hybrid",
    "patch_ms_scales": (8, 16, 32),
    "patch_edge_k": 4,
    "patch_refine_layers": 2,
    "patch_refine_k": 8,
    "disable_patch_refiner": False,
}


def parse_patch_ms_scales(value):
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return tuple(DEFAULT_BACKBONE_CONFIG["patch_ms_scales"])
        value = tuple(int(p) for p in parts)
    elif isinstance(value, int):
        value = (int(value),)
    elif isinstance(value, (list, tuple)):
        value = tuple(int(v) for v in value)
    else:
        value = tuple(DEFAULT_BACKBONE_CONFIG["patch_ms_scales"])
    clean = tuple(sorted({max(1, int(v)) for v in value}))
    return clean or tuple(DEFAULT_BACKBONE_CONFIG["patch_ms_scales"])


def add_backbone_args(parser):
    parser.add_argument("--num_group", type=int, default=DEFAULT_BACKBONE_CONFIG["num_group"])
    parser.add_argument("--group_size", type=int, default=DEFAULT_BACKBONE_CONFIG["group_size"])
    parser.add_argument(
        "--patch_encoder_type",
        type=str,
        default=DEFAULT_BACKBONE_CONFIG["patch_encoder_type"],
        choices=["pointnet", "hybrid"],
    )
    parser.add_argument(
        "--patch_ms_scales",
        type=str,
        default="8,16,32",
        help="Comma-separated patch-internal scales used by the EdgeConv branch.",
    )
    parser.add_argument("--patch_edge_k", type=int, default=DEFAULT_BACKBONE_CONFIG["patch_edge_k"])
    parser.add_argument("--patch_refine_layers", type=int, default=DEFAULT_BACKBONE_CONFIG["patch_refine_layers"])
    parser.add_argument("--patch_refine_k", type=int, default=DEFAULT_BACKBONE_CONFIG["patch_refine_k"])
    parser.add_argument("--disable_patch_refiner", action="store_true", default=False)
    return parser


def build_backbone_config(args=None, **overrides):
    cfg = dict(DEFAULT_BACKBONE_CONFIG)
    if args is not None:
        for key in cfg:
            if hasattr(args, key):
                cfg[key] = getattr(args, key)
    cfg.update(overrides)
    cfg["patch_encoder_type"] = str(cfg["patch_encoder_type"]).strip().lower()
    if cfg["patch_encoder_type"] not in ("pointnet", "hybrid"):
        raise ValueError(f"Unsupported patch encoder type: {cfg['patch_encoder_type']}")
    cfg["patch_ms_scales"] = parse_patch_ms_scales(cfg["patch_ms_scales"])
    cfg["patch_edge_k"] = max(1, int(cfg["patch_edge_k"]))
    cfg["patch_refine_layers"] = max(0, int(cfg["patch_refine_layers"]))
    cfg["patch_refine_k"] = max(1, int(cfg["patch_refine_k"]))
    cfg["disable_patch_refiner"] = bool(cfg["disable_patch_refiner"])
    cfg["num_group"] = int(cfg["num_group"])
    cfg["group_size"] = int(cfg["group_size"])
    cfg["color"] = bool(cfg["color"])
    return EasyDict(cfg)
