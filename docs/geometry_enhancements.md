# PatchAlign3D 几何增强说明

## 改进动机
原始 PatchAlign3D backbone 结构简洁、效果稳定，但在局部几何建模方面有两个比较明显的限制：

1. 在每个 patch 内部，编码器本质上是 PointNet 风格。这样的做法效率高、实现简单，但更多是把 patch 当成无序点集来处理，没有显式建模相邻点之间的局部几何关系。
2. 在 patch 与 patch 之间，模型主要依赖全局 transformer 做信息交互。它确实能够建模全局上下文，但在进入全局建模之前，没有专门突出相邻 patch center 之间的短程几何关系。

从本科毕设改进的角度看，这使得 PatchAlign3D 很适合做一次“patch 内局部几何增强 + patch 间局部几何增强”的升级，同时又能保持原始两阶段训练框架不变。

## 新增模块
### 1. HybridPatchEncoder
我们保留原始的 PointNet patch encoder，同时额外增加一个并行的多尺度 EdgeConv 分支。

- PointNet 分支保留了原始 PatchAlign3D 的 patch token 提取流程。
- 多尺度 EdgeConv 分支借鉴了 DGCNN 和 PointNet++ 的局部几何建模思想。
- 它对同一个 patch 在三个尺度上进行处理，分别使用 8、16、32 个点。
- 每个尺度内部都会构建 patch 内部的 kNN 图，并经过两层轻量的 EdgeConv 单元。
- 三个尺度得到的描述会先融合，再与 PointNet 分支的描述拼接融合。

这样的设计可以在保留原始 PointNet 分支稳定性的同时，增强模型对局部几何结构和细粒度部件边界的感知能力。

### 2. PatchGeometryRefiner
在 patch token 进入原始全局 transformer 之前，我们增加了一个轻量的 patch 级几何 refinement 模块。

- 它首先在 patch center 之间建立局部邻域图。
- 每个 block 都会利用相邻 patch 之间的相对位置编码。
- 邻域消息先在局部范围内聚合，然后再交给后续的全局 transformer 做全局建模。

这个模块的设计思路主要借鉴了 Point Transformer 和 Point Transformer V2 中“相对几何位置对局部关系建模很重要”的思想。

## 文献依据
这次改进主要参考了点云领域中几类成熟且流行的方法：

- PointNet++：强调多尺度局部结构建模。
  论文链接：https://papers.nips.cc/paper_files/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html
- DGCNN / EdgeConv：强调在点邻域图上显式建模局部几何关系。
  论文链接：https://arxiv.org/abs/1801.07829
- Point Transformer：强调基于相对位置编码的局部关系建模。
  论文链接：https://arxiv.org/abs/2012.09164
- Point Transformer V2：进一步增强局部几何注意力与位置编码建模。
  论文链接：https://arxiv.org/abs/2210.05666
- Point-TnT：强调 patch 内与 patch 间的分层建模思路，与本次“patch 内增强 + patch 间增强”的设计思路相近。
  论文链接：https://arxiv.org/abs/2204.03957

其中，各新增模块与参考文献的对应关系可以概括为：

- `HybridPatchEncoder` 里的多尺度分支主要参考了 PointNet++ 的多尺度建模思想，以及 DGCNN / EdgeConv 的局部邻域特征聚合方式。
- `PatchGeometryRefiner` 主要参考了 Point Transformer 和 Point Transformer V2 中基于相对位置编码的局部关系建模方式。
- 整体“patch 内建模 + patch 间建模”的结构组织方式，也借鉴了 Point-TnT 这类分层建模方法的思路。

整体实现刻意保持轻量，目标是尽量兼容当前 PatchAlign3D 的代码框架和训练流程，同时避免引入新的自定义 CUDA 依赖，降低环境配置风险。

## 保持不变的部分
虽然 backbone 做了几何增强，但 PatchAlign3D 的以下核心设定保持不变：

- 两阶段训练流程保持不变。
- Stage 1 仍然使用与离线 DINO patch 特征的余弦相似度回归。
- Stage 2 仍然执行 patch token 与文本特征的对齐。
- 训练数据格式以及 `patch_dino/patch_features.pt` 缓存格式保持不变。
- `forward_patches()` 的输出接口保持不变，仍然返回 patch embedding、patch center 和 patch 成员索引，shape 也与原实现一致。

## 可用于消融实验的版本
当前实现支持比较清晰的消融设置，便于后续论文撰写和答辩展示：

- 仅使用原始 PointNet patch encoder。
- PointNet patch encoder + PatchGeometryRefiner。
- HybridPatchEncoder。
- HybridPatchEncoder + PatchGeometryRefiner。

这些变体都可以通过命令行中的 backbone 相关参数进行切换。
