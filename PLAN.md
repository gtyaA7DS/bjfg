# PatchAlign3D 主体升级方案：双模块几何增强版

## Summary
在不改两阶段训练范式、不改 `Stage 1` 余弦相似度 loss、不改数据格式与 `patch_dino` 缓存格式的前提下，对 backbone 做一次“本科毕设可见工作量”的升级，目标是增强 patch 内局部几何建模和 patch 间几何关系建模，同时保证 `stage1` 先稳定跑通。

这次改动采用“双模块增强”路线，并额外补一份简短设计说明，方便后续写毕设与答辩。

文献依据只用成熟、流行、且和当前架构兼容的方向：
- PointNet++：多尺度局部结构建模  
  https://papers.nips.cc/paper_files/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html
- DGCNN / EdgeConv：动态图局部邻域特征聚合  
  https://arxiv.org/abs/1801.07829
- Point Transformer：相对位置感知的点云注意力  
  https://arxiv.org/abs/2012.09164
- Point Transformer V2：更强的局部几何注意力与位置编码  
  https://arxiv.org/abs/2210.05666
- Point-TnT：patch 内与 patch 间分层建模  
  https://arxiv.org/abs/2204.03957

当前环境已确认可用：`/root/data/core_subset` 存在、`patch_dino` 缓存存在、`patchalign3d` conda 环境可导入 `torch/open_clip/timm/transformers/pointnet2_ops/knn_cuda`，并且现有 `stage1` 已能进入训练循环。正式测试目录统一按当前仓库 `/root/bjfg_new` 处理，不再使用旧的 `PYTHONPATH=/root/bjfg`。

## Key Changes
### 1. 主干模型升级
在 [point_transformer.py](/root/bjfg_new/src/models/point_transformer.py) 上做两处结构增强，输出接口保持不变，仍然返回 `patch_emb / patch_centers / patch_idx`。

- 新增 `HybridPatchEncoder`，替换当前单一 `PointNet` patch encoder。
- `HybridPatchEncoder` 保留原始 PointNet 分支，作为稳定基线分支。
- 同时新增一个多尺度 `EdgeConv` 分支，作为局部几何增强分支。
- 多尺度分支不改外部 patch 划分逻辑；仍使用同一个 patch，只在 patch 内按到中心点距离排序后取 3 个子尺度：`[8, 16, 32]`。
- 每个尺度内部构建 patch 内小图，默认内部邻居数 `edge_k=4`，做轻量 `EdgeConv` 聚合。
- PointNet 分支与多尺度 EdgeConv 分支输出拼接后，经一个轻量融合 MLP 投到原来的 `encoder_dims`，保证后续 transformer 和 loss 不需要改 shape。

- 新增 `PatchGeometryRefiner`，放在 patch token 进入全局 transformer 之前。
- `PatchGeometryRefiner` 只在 patch center 上建图，不改点级数据流。
- 默认在 patch center 的 XYZ 空间做 `k=8` 的 patch-level kNN。
- 每个 refiner block 使用 “邻居 token 差分 + 相对坐标编码 + residual” 的局部聚合，风格接近 Point Transformer / PTv2，但保持实现轻量，不引入额外 CUDA 依赖。
- 默认堆叠 `2` 个 refiner block，然后再进入原有全局 transformer。
- 原有 transformer 仍保留，作为全局上下文模块；这次不是替换，而是“局部几何 refinement + 全局 transformer” 串联。

### 2. 统一两阶段接口与配置
- 新建统一的 backbone config/helper，避免 `stage1 / stage2 / infer / eval` 各自硬编码模型超参。
- 默认把 `depth` 统一为 `12`，与论文描述和现有 `stage2/infer` 保持一致，避免后续 `Stage 1 -> Stage 2` 继续依赖不完整权重加载。
- 新增可控开关，支持后续答辩/论文做 ablation：
  - `--patch_encoder_type pointnet|hybrid`，默认 `hybrid`
  - `--patch_ms_scales 8,16,32`
  - `--patch_edge_k 4`
  - `--patch_refine_layers 2`
  - `--patch_refine_k 8`
  - `--disable_patch_refiner`，默认关闭时不传，默认启用
- `Stage 1`、`Stage 2`、推理和评测入口都要接收并透传这些参数，但 loss、数据流和输出行为不改。
- `Stage 1` 仍然是在线 patch center 对齐离线 cached patch center，再做余弦回归。
- `Stage 2` 仍然是 patch-text BCE with logits；这次不跑 full stage2，只保证接口兼容和脚本可正常构建模型。

### 3. 成果可见化
- 新增一份简短设计说明文档到 `docs/`，并在 [README.md](/root/bjfg_new/README.md) 加入口。
- 文档内容固定包括：
  - 为什么原始 PatchAlign3D 适合做 patch 内/patch 间双增强
  - 每个新增模块对应哪篇文献、借鉴了什么
  - 和原版相比，保持了哪些不变项：两阶段训练、Stage1 cosine loss、数据集与缓存格式
  - 为后续毕设撰写预留的 ablation 维度：原版、只加 HybridPatchEncoder、只加 PatchGeometryRefiner、双模块全开

## Test Plan
### 1. 静态与接口检查
- `python -m compileall src`
- 用当前环境实例化新模型，确认 `forward_patches()` 输出 shape 不变。
- 跑 `stage2`、`infer`、`eval` 的 `--help` 或最小 import，确认新参数已串通，且未破坏两阶段入口。

### 2. Stage 1 快速 smoke test
先做一次超小配置的功能测试，只验证新模块前向、反向、loss、日志、checkpoint 路径都正常：
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

### 3. 按你的要求做 Stage 1 小参数实测
正式测试命令按你要求执行，但路径修正为当前仓库：
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

固定的回退策略：
- 如果显存不足，只允许把 `batch_size` 按 `8 -> 4 -> 2` 递减。
- `npoint / num_group / group_size / loss / 新模块结构` 都不改。
- 最终记录实际跑通的 batch size 和对应日志目录。

验收标准：
- 脚本能完成至少 1 个 epoch。
- `train loss` 为有限值，不出现 `nan/inf`。
- 能生成日志目录与 checkpoint。
- 新 backbone 在 `stage1` 下无 shape mismatch、无参数缺失导致的运行时错误。

## Assumptions
- 不新增新的第三方 CUDA 扩展，优先只用现有 `torch + pointnet2_ops + knn_cuda`，降低环境风险。
- 数据集、采样方式、`patch_dino/patch_features.pt` 结构、Stage1 对齐逻辑保持不变。
- 本轮只做 `stage1` 实跑验证；`stage2` 只做接口兼容检查，不跑完整训练。
- 为了两阶段后续可用性，这次会顺手统一默认 backbone depth 为 `12`。
- 这次改动默认以“代码 + 设计说明”交付，强调毕设可见工作量，而不是只做最小修 bug。
