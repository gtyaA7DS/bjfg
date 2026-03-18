import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

from patchalign3d.models.config import build_backbone_config


def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data


def _gather_neighbors(x, idx):
    batch_size, num_points, channels = x.shape
    _, query_points, k = idx.shape
    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
    flat_idx = (idx + idx_base).reshape(-1)
    flat_x = x.reshape(batch_size * num_points, channels)
    gathered = flat_x[flat_idx]
    return gathered.view(batch_size, query_points, k, channels)


def _build_knn_indices(coords, k, exclude_self=True):
    batch_size, num_points, _ = coords.shape
    if num_points <= 1:
        return torch.zeros(batch_size, num_points, 1, dtype=torch.long, device=coords.device)
    max_neighbors = num_points - 1 if exclude_self else num_points
    k_eff = min(max(1, int(k)), max_neighbors)
    dist = torch.cdist(coords.float(), coords.float())
    if exclude_self:
        eye = torch.eye(num_points, device=coords.device, dtype=dist.dtype).unsqueeze(0)
        dist = dist + eye * 1e6
    return dist.topk(k=k_eff, dim=-1, largest=False).indices


class Group(nn.Module):

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, channels = xyz.shape
        if channels > 3:
            xyz_only = xyz[:, :, :3].contiguous()
            extra = xyz[:, :, 3:].contiguous()
        else:
            xyz_only = xyz.contiguous()
            extra = None
        center = fps(xyz_only, self.num_group)
        _, idx = self.knn(xyz_only, center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_flat = (idx + idx_base).view(-1)
        neigh_xyz = xyz_only.view(batch_size * num_points, -1)[idx_flat, :].view(
            batch_size, self.num_group, self.group_size, 3
        )
        if extra is not None:
            neigh_extra = extra.view(batch_size * num_points, -1)[idx_flat, :].view(
                batch_size, self.num_group, self.group_size, -1
            )
            neighborhood = torch.cat((neigh_xyz - center.unsqueeze(2), neigh_extra), dim=-1)
        else:
            neighborhood = neigh_xyz - center.unsqueeze(2)
        return neighborhood, center


class PatchedGroup(nn.Module):
    """Same as Group but also returns patch membership indices."""

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, channels = xyz.shape
        if channels > 3:
            xyz_only = xyz[:, :, :3].contiguous()
            extra = xyz[:, :, 3:].contiguous()
        else:
            xyz_only = xyz.contiguous()
            extra = None

        center = fps(xyz_only, self.num_group)
        _, idx = self.knn(xyz_only, center)
        idx_rel = idx.clone()
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_flat = (idx + idx_base).view(-1)
        neigh_xyz = xyz_only.view(batch_size * num_points, -1)[idx_flat, :].view(
            batch_size, self.num_group, self.group_size, 3
        )

        if extra is not None:
            neigh_extra = extra.view(batch_size * num_points, -1)[idx_flat, :].view(
                batch_size, self.num_group, self.group_size, -1
            )
            neighborhood = torch.cat((neigh_xyz - center.unsqueeze(2), neigh_extra), dim=-1)
        else:
            neighborhood = neigh_xyz - center.unsqueeze(2)
        return neighborhood.contiguous(), center.contiguous(), idx_rel


class PointNetBranch(nn.Module):

    def __init__(self, encoder_channel, color=False):
        super().__init__()
        in_dim = 6 if color else 3
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        batch_size, num_group, num_points, channels = point_groups.shape
        point_groups = point_groups.reshape(batch_size * num_group, num_points, channels).permute(0, 2, 1)
        feature = self.first_conv(point_groups)
        feature_global = torch.max(feature, 2, keepdim=True)[0]
        feature_global = feature_global.repeat(1, 1, num_points)
        feature = torch.cat([feature_global, feature], 1)
        feature = self.second_conv(feature)
        feature = feature.max(dim=2)[0]
        feature = feature.reshape(batch_size, num_group, self.encoder_channel).contiguous()
        return feature


class EdgeConvUnit(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.k = int(k)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, feats, coords):
        if feats.shape[1] <= 1:
            return self.edge_mlp(torch.cat([feats, torch.zeros_like(feats), torch.zeros_like(coords)], dim=-1))
        idx = _build_knn_indices(coords, self.k, exclude_self=True)
        neighbor_feats = _gather_neighbors(feats, idx)
        neighbor_coords = _gather_neighbors(coords, idx)
        center_feats = feats.unsqueeze(2).expand(-1, -1, idx.size(-1), -1)
        center_coords = coords.unsqueeze(2).expand(-1, -1, idx.size(-1), -1)
        edge_feat = torch.cat(
            [center_feats, neighbor_feats - center_feats, neighbor_coords - center_coords], dim=-1
        )
        edge_feat = self.edge_mlp(edge_feat)
        return edge_feat.max(dim=2).values


class EdgeConvScale(nn.Module):
    def __init__(self, in_dim, hidden_dim, k):
        super().__init__()
        self.unit1 = EdgeConvUnit(in_dim=in_dim, out_dim=hidden_dim, k=k)
        self.unit2 = EdgeConvUnit(in_dim=hidden_dim, out_dim=hidden_dim, k=k)

    def forward(self, feats, coords):
        x = self.unit1(feats, coords)
        x = self.unit2(x, coords)
        return x.max(dim=1).values


class MultiScaleEdgeConvBranch(nn.Module):
    def __init__(self, in_dim, scales, edge_k, out_dim=256, hidden_dim=128):
        super().__init__()
        self.scales = tuple(int(s) for s in scales)
        self.scale_blocks = nn.ModuleList(
            [EdgeConvScale(in_dim=in_dim, hidden_dim=hidden_dim, k=edge_k) for _ in self.scales]
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * len(self.scales), out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, point_groups):
        batch_size, num_group, num_points, channels = point_groups.shape
        points = point_groups.reshape(batch_size * num_group, num_points, channels)
        coords = points[:, :, :3]
        radius = coords.norm(dim=-1)
        order = radius.argsort(dim=-1)
        points = points.gather(1, order.unsqueeze(-1).expand(-1, -1, channels))
        coords = coords.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))

        scale_features = []
        for scale, block in zip(self.scales, self.scale_blocks):
            scale_n = min(int(scale), num_points)
            scale_points = points[:, :scale_n, :]
            scale_coords = coords[:, :scale_n, :]
            scale_features.append(block(scale_points, scale_coords))

        fused = torch.cat(scale_features, dim=-1)
        fused = self.fusion_mlp(fused)
        return fused.reshape(batch_size, num_group, self.out_dim).contiguous()


class HybridPatchEncoder(nn.Module):
    def __init__(self, encoder_channel, color=False, patch_ms_scales=(8, 16, 32), patch_edge_k=4):
        super().__init__()
        in_dim = 6 if color else 3
        self.pointnet_branch = PointNetBranch(encoder_channel=encoder_channel, color=color)
        self.edge_branch = MultiScaleEdgeConvBranch(
            in_dim=in_dim,
            scales=patch_ms_scales,
            edge_k=patch_edge_k,
            out_dim=256,
            hidden_dim=128,
        )
        self.final_fuse = nn.Sequential(
            nn.Linear(encoder_channel + 256, encoder_channel),
            nn.GELU(),
            nn.Linear(encoder_channel, encoder_channel),
        )

    def forward(self, point_groups):
        pointnet_feat = self.pointnet_branch(point_groups)
        edge_feat = self.edge_branch(point_groups)
        fused = torch.cat([pointnet_feat, edge_feat], dim=-1)
        return self.final_fuse(fused)


class Encoder(nn.Module):

    def __init__(
        self,
        encoder_channel,
        color=False,
        patch_encoder_type="hybrid",
        patch_ms_scales=(8, 16, 32),
        patch_edge_k=4,
    ):
        super().__init__()
        self.patch_encoder_type = str(patch_encoder_type).lower()
        if self.patch_encoder_type == "hybrid":
            self.impl = HybridPatchEncoder(
                encoder_channel=encoder_channel,
                color=color,
                patch_ms_scales=patch_ms_scales,
                patch_edge_k=patch_edge_k,
            )
        else:
            self.impl = PointNetBranch(encoder_channel=encoder_channel, color=color)

    def forward(self, point_groups):
        return self.impl(point_groups)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RefinerBlock(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.k = int(k)
        self.norm1 = nn.LayerNorm(dim)
        self.pos_mlp = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.msg_mlp = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(in_features=dim, hidden_features=dim * 4, act_layer=nn.GELU, drop=0.0)

    def forward(self, tokens, centers):
        tokens_norm = self.norm1(tokens)
        idx = _build_knn_indices(centers, self.k, exclude_self=True)
        neighbor_tokens = _gather_neighbors(tokens_norm, idx)
        neighbor_centers = _gather_neighbors(centers, idx)
        center_tokens = tokens_norm.unsqueeze(2).expand(-1, -1, idx.size(-1), -1)
        center_xyz = centers.unsqueeze(2).expand(-1, -1, idx.size(-1), -1)
        pos_feat = self.pos_mlp(neighbor_centers - center_xyz)
        msg = self.msg_mlp(torch.cat([center_tokens, neighbor_tokens - center_tokens, pos_feat], dim=-1))
        tokens = tokens + msg.mean(dim=2)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class PatchGeometryRefiner(nn.Module):
    def __init__(self, dim, num_layers, k):
        super().__init__()
        self.blocks = nn.ModuleList([RefinerBlock(dim=dim, k=k) for _ in range(num_layers)])

    def forward(self, tokens, centers):
        for block in self.blocks:
            tokens = block(tokens, centers)
        return tokens


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        def _drop_for_block(i):
            if isinstance(drop_path_rate, (list, tuple)):
                return drop_path_rate[i]
            return drop_path_rate

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=_drop_for_block(i),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for blk in self.blocks:
            x = blk(x + pos)
        return x


class get_model(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = build_backbone_config(config)
        self.trans_dim = self.config.trans_dim
        self.depth = self.config.depth
        self.drop_path_rate = self.config.drop_path_rate
        self.num_heads = self.config.num_heads
        self.color = getattr(self.config, "color", False)
        self.group_size = self.config.group_size
        self.num_group = self.config.num_group
        self.encoder_dims = self.config.encoder_dims

        self.group_divider = PatchedGroup(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(
            encoder_channel=self.encoder_dims,
            color=self.color,
            patch_encoder_type=self.config.patch_encoder_type,
            patch_ms_scales=self.config.patch_ms_scales,
            patch_edge_k=self.config.patch_edge_k,
        )
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        if self.config.disable_patch_refiner or self.config.patch_refine_layers <= 0:
            self.patch_refiner = None
        else:
            self.patch_refiner = PatchGeometryRefiner(
                dim=self.trans_dim,
                num_layers=self.config.patch_refine_layers,
                k=self.config.patch_refine_k,
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_patches(self, pts):
        """
        Args:
            pts: (B, C, N) with C>=3 (xyz | [extra])
        Returns:
            patch_emb    : (B, trans_dim, G)
            patch_centers: (B, 3, G)
            patch_idx    : (B, G, M)
        """
        pts_bn = pts.transpose(-1, -2).contiguous()
        neighborhood, center, patch_idx = self.group_divider(pts_bn)
        group_tokens = self.encoder(neighborhood)
        group_tokens = self.reduce_dim(group_tokens)
        if self.patch_refiner is not None:
            group_tokens = self.patch_refiner(group_tokens, center)

        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        feature = self.blocks(x, pos)
        patch_emb = self.norm(feature)[:, 1:, :].transpose(-1, -2).contiguous()
        patch_centers = center.transpose(-1, -2).contiguous()
        return patch_emb, patch_centers, patch_idx

    def forward(self, pts):
        patch_emb, _, _ = self.forward_patches(pts)
        return patch_emb
