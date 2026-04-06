# PatchAlign3D 几何增强版执行计划

## Summary
在保持两阶段训练流程、`Stage 1` 余弦相似度回归 loss、训练数据与 `patch_dino` 缓存格式不变的前提下，对 backbone 做一次“双模块增强”升级：

- `HybridPatchEncoder`：在原始 PointNet patch encoder 旁边新增一个多尺度 `EdgeConv` 分支，强化 patch 内局部几何建模。
- `PatchGeometryRefiner`：在 patch token 进入全局 transformer 之前，增加基于 patch center 邻域的局部几何 refinement，强化 patch 间几何关系建模。

这次改动的默认目标是“改完仍能直接跑现有 `stage1` 入口”，并且保留足够的开关以支持毕设中的 ablation。测试策略按你刚确认的约束执行：**不碰你当前 GPU 0 上已有训练，只做不干扰它的极小型 smoke test；如果显存不够，就退化为代码实现 + 静态检查，不强行抢占 GPU。**

## Implementation Changes
### 1. Backbone 主体升级
在模型主干中保留原 patch 划分、token 生成和 transformer 主流程，只替换 patch encoder 并插入 patch-level refinement。

- 新增 `PointNetBranch`，把当前 `Encoder` 的 PointNet 逻辑拆成独立分支，功能与原实现保持一致。
- 新增 `MultiScaleEdgeConvBranch`，作为并行局部几何分支。
- `MultiScaleEdgeConvBranch` 输入仍为单个 patch 的局部点集 `(B,G,M,C)`；图构建只基于前 3 维相对坐标，若有额外通道则参与特征编码但不参与建图。
- 在每个 patch 内先按点到 patch center 的半径排序，再取三种固定尺度子集：`[8,16,32]`。
- 每个尺度内使用 `torch.cdist + topk` 建 patch 内 kNN 图，默认 `edge_k=4`，不新增 CUDA 依赖。
- 每个尺度使用两层轻量 `EdgeConvUnit`：
  - 第一层输入为 `[f_i, f_j - f_i, p_j - p_i]`
  - 聚合方式固定为 `max` over neighbors
  - 每层输出通道固定为 `128`
- 每个尺度最终对点维做 `max pool` 得到一个 `128` 维尺度描述。
- 三个尺度描述拼接后，经 `fusion_mlp` 投到 `256` 维。
- `PointNetBranch` 输出 `256` 维，`MultiScaleEdgeConvBranch` 输出 `256` 维，两者拼接后经 `final_fuse` 回到原始 `encoder_dims`，从而保证后续接口和 head 不变。
- 默认新的 patch encoder 名称为 `HybridPatchEncoder`，并作为默认编码器；保留 `pointnet-only` 模式用于对比实验。

### 2. Patch 间几何 refinement
- 在 patch token 进入原全局 transformer 之前，新增 `PatchGeometryRefiner`。
- `PatchGeometryRefiner` 工作在 `reduce_dim` 之后的 token 空间，输入为 `(B,G,trans_dim)` 和 patch centers `(B,G,3)`。
- 默认堆叠 `2` 个 `RefinerBlock`。
- 每个 `RefinerBlock` 只在 patch center 上做局部建图，默认 `refine_k=8`。
- 每个 block 采用固定结构：
  - `LayerNorm` 当前 token
  - 基于 patch center 的 kNN 邻域索引
  - 相对位置编码 `pos_mlp: 3 -> trans_dim`
  - 消息函数 `msg_mlp([x_i, x_j - x_i, pos_ij]) -> trans_dim`
  - 对邻域做 `mean` 聚合
  - residual 更新
  - 再接一个标准前馈 `ffn` residual
- `RefinerBlock` 不引入 cls token；cls token 仍按当前实现只在进入全局 transformer 时加入。
- 原有 transformer blocks 保留，不替换；最终结构固定为：
  `PatchedGroup -> HybridPatchEncoder -> reduce_dim -> PatchGeometryRefiner -> add cls/pos -> global transformer`

### 3. 配置与入口统一
所有训练、推理、评测入口统一接收同一组 backbone 参数，避免 `stage1/stage2/infer/eval` 模型定义分叉。

- 新增统一 backbone config builder，默认 `depth=12`，统一到论文和现有 `stage2/infer/eval` 的配置，消除当前 `stage1=15`、`stage2=12` 的不一致。
- 新增 CLI 参数，所有入口都要支持并透传：
  - `--patch_encoder_type pointnet|hybrid`，默认 `hybrid`
  - `--patch_ms_scales 8,16,32`
  - `--patch_edge_k 4`
  - `--patch_refine_layers 2`
  - `--patch_refine_k 8`
  - `--disable_patch_refiner`，默认不传即启用
- `Stage 1`、`Stage 2`、`infer`、`eval` 的输出接口和调用方式保持不变。
- `Stage 1` 的 patch-center 对齐逻辑、DINO 目标缓存读取、cosine regression loss 不改。
- `Stage 2` 的 patch-text BCE、温度参数、文本 cache/bank 逻辑不改，只做模型构建兼容。
- README 中所有测试命令统一修正为真实仓库路径 `PYTHONPATH=/root/bjfg_new`。

### 4. 文献驱动设计说明
新增一份简短设计说明，作为毕设成果呈现的一部分。

文档必须固定包含：
- 原始 PatchAlign3D 的局限：patch 内仅 PointNet、patch 间仅全局 self-attention。
- 为什么选多尺度 `EdgeConv`：借鉴 DGCNN、PointNet++ 的局部几何建模思路，适合 patch 内细节增强。
- 为什么选 patch-level refiner：借鉴 Point Transformer / PTv2 的相对位置建模，适合 patch 间几何关系增强。
- 与原版保持不变的内容：两阶段训练、Stage1 cosine loss、数据与缓存格式。
- 可直接用于答辩的 ablation 版本：
  - 原版 PointNet encoder
  - 只加 `HybridPatchEncoder`
  - 只加 `PatchGeometryRefiner`
  - 双模块全开

## Public Interfaces
### Backbone 行为
- `forward_patches()` 的返回值保持不变：
  - `patch_emb: (B, trans_dim, G)`
  - `patch_centers: (B, 3, G)`
  - `patch_idx: (B, G, M)`
- 现有 `proj`、`stage1` 训练循环、`stage2` 文本 head、推理和评测代码都不需要改 tensor contract。

### 新增命令行参数
这些参数必须同时出现在训练、推理和评测入口中，并在保存 checkpoint 时写入 `args`：
- `patch_encoder_type`
- `patch_ms_scales`
- `patch_edge_k`
- `patch_refine_layers`
- `patch_refine_k`
- `disable_patch_refiner`

## Test Plan
### 1. 静态与构建检查
- `python -m compileall src`
- 在 `patchalign3d` conda 环境中做最小 import：
  - `stage1`
  - `stage2`
  - `infer`
  - `eval`
- 实例化默认 `hybrid` 模型与 `pointnet-only` 模型各一次，确认：
  - 模型可构建
  - `forward_patches()` shape 不变
  - 参数能被 optimizer 正常收集

### 2. 无 GPU 抢占的功能检查
在不影响你当前 GPU 训练的前提下，先做静态前向检查：
- 从训练集取 1 个 batch
- 构造默认 `hybrid` 模型
- 完成一次 `model.forward_patches()`、`proj()`、`stage1` 目标对齐与 loss 计算
- 不跑完整 epoch，只验证前向/反向/梯度存在、无 shape mismatch、无 `nan/inf`

### 3. 极小型 Stage1 smoke test
只有在当前 GPU 显存允许且不干扰你现有训练时，才执行以下 smoke test：
```bash
source /home/vipuser/anaconda3/etc/profile.d/conda.sh
conda activate patchalign3d
PYTHONPATH=/root/bjfg_new python src/training/stage1.py \
  --data_root /root/data/core_subset \
  --train_list /root/data/core_subset/labeled/split/train.txt \
  --val_list /root/data/core_subset/labeled/split/val.txt \
  --gpu 0 --batch_size 2 --epoch 1 \
  --eval_every 0 --save_every 0 \
  --npoint 1024 --num_group 32 --group_size 16 \
  --random_sample_train --train_encoder \
  --dino_feature_subdir patch_dino \
  --workers 0 --wandb_mode disabled
```

### 4. 按你要求的 Stage1 小参数实测
只有在 smoke test 成功且不会影响你现有训练时，再尝试：
```bash
source /home/vipuser/anaconda3/etc/profile.d/conda.sh
conda activate patchalign3d
PYTHONPATH=/root/bjfg_new python src/training/stage1.py \
  --data_root /root/data/core_subset \
  --train_list /root/data/core_subset/labeled/split/train.txt \
  --val_list /root/data/core_subset/labeled/split/val.txt \
  --gpu 0 --batch_size 8 --epoch 1 \
  --eval_every 5 --save_every 5 \
  --npoint 5000 --num_group 128 --group_size 32 \
  --random_sample_train --train_encoder \
  --dino_feature_subdir patch_dino \
  --workers 0 --wandb_mode disabled
```

显存/资源回退策略固定为：
- 不停止你现有训练。
- 如果资源不足，只允许将 `batch_size` 从 `8 -> 4 -> 2` 递减。
- 不改 `npoint / num_group / group_size / 新模块结构 / loss`。
- 若仍无法并行测试，则本轮验收退化为“代码实现 + 静态检查 + 单 batch 前后向成功”。

### 5. 验收标准
实现完成后至少满足以下条件：
- 默认 `hybrid` 模型可被 `stage1/stage2/infer/eval` 正常构建。
- `Stage 1` loss 仍为余弦回归，且值为有限数。
- `Stage 1` 新模型在最小前后向检查中无 `shape mismatch`、无 `nan/inf`。
- 若显存允许，完成至少一次不干扰现有训练的 `stage1` smoke test。
- README 和设计说明能清楚体现“新增了两类几何增强模块”和“为什么这样设计”。

## Assumptions
- 当前真实仓库路径是 `/root/bjfg_new`，不是旧的 `/root/bjfg`。
- `patchalign3d` conda 环境和 `patch_dino` 数据缓存已可用。
- 不新增新的外部 CUDA 扩展；所有新模块只使用现有依赖。
- 本轮只要求实跑 `stage1`；`stage2` 只做兼容性构建检查。
- 不干扰你现在 GPU 0 上已经在跑的训练进程。
