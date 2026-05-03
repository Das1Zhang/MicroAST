# 时序一致性改进 (Temporal Consistency)

## 1. 问题背景：视频帧的“时序灾难” (Temporal Inconsistency)

MicroAST 是一个纯 2D 图像模型，没有任何时序记忆或帧间约束。当直接逐帧处理视频时：

```
Frame t:    → MicroAST → Stylized t   (某个局部纹理为蓝色)
Frame t+1:  → MicroAST → Stylized t+1 (同一位置的纹理突然变为红色)
```

即使相邻两帧内容几乎相同（仅微小位移），模型对每一帧完全独立计算的特征和渲染的局部纹理也会发生剧烈变化，导致**严重的闪烁 (flickering)**。

**根因分析**：

- **内容编码器的帧间抖动**：相邻帧经过 content encoder 后产生微小差异的特征，但 decoder 的非线性处理会将这种微小差异放大为可见的纹理变化
- **全局调制的无状态性**：每帧独立计算 AdaIN 参数和 FilterMod 权重，缺乏跨帧的平滑约束

## 2. 改进方案总览

本次改进分为两个互补的部分：

| 方案 | 阶段 | 成本 | 作用 |
|------|------|------|------|
| EMA 时序平滑 | 推理时 | **零成本**（几次张量加减乘） | 在 latent space 平滑帧间信号，即时抑制闪烁 |
| 光流时序损失 | 训练时 | 较高（需要光流网络 + 视频数据集） | 从权重层面学习时序一致性，治本 |

## 3. Part 1: EMA 时序平滑（推理时）

### 设计原则

**在 latent space 而非 pixel space 平滑**。如果直接在像素空间做 EMA（`output = 0.7*prev_output + 0.3*curr_output`），运动会造成严重的 ghosting（拖影）。但在 latent space（内容特征、调制信号）做 EMA，能有效抑制 jitter 而不引入 ghosting。

### 3.1 `TemporalSmoother` 类（`temporal_smoother.py`）

向三个信号流分别建立 EMA 缓存：

```
                         ┌─ smooth_style(style_feats) ──────────→ s (smoothed)
                         │
Style → StyleEncoder ────┤
                         └─→ Modulator → smooth_modulation(w,b) → w,b (smoothed)

Content → ContentEncoder → smooth_content(content_feats) ──────→ x (smoothed)
                                                                  │
                                        Decoder ←─────────────────┘
```

**核心实现**：

```python
class TemporalSmoother:
    def __init__(self, momentum=0.7):
        self.momentum = momentum
        # 四个缓存（prev_*）
        self.prev_content_feats = None
        self.prev_style_feats = None
        self.prev_weights = None
        self.prev_biases = None

    def smooth_content(self, content_feats):
        # blended = 0.7 * prev_feats + 0.3 * curr_feats
        # 内容特征帧间差异是闪烁的主要来源，平滑此信号效果最显著
        ...

    def smooth_style(self, style_feats):
        # 当 style image 不变时，第一帧后即为 no-op
        # 当 style 渐变时（如多风格切换），提供平滑过渡
        ...

    def smooth_modulation(self, weights, biases):
        # 抑制调制参数的微观抖动
        ...
```

**三路平滑的目标和效果**：

| 平滑目标 | 主要闪烁来源 | 平滑效果 |
|----------|-------------|---------|
| `smooth_content` | **是**（内容特征的微小差异被 decoder 放大） | 效果最显著 |
| `smooth_style` | 否（固定 style 时不变） | 渐进风格切换时平滑过渡 |
| `smooth_modulation` | 轻微（数值精度差异） | 稳定性增强 |

**参数 `momentum` 调优指南**：

| momentum | 效果 | 适用场景 |
|----------|------|---------|
| 0.5 | 轻度平滑，响应快 | 高帧率、快速运动 |
| 0.7 | **推荐默认值** | 大多数视频 |
| 0.9 | 强平滑，响应慢 | 静态镜头、慢速运动 |

### 3.2 `VideoTestNet` 类（`net_microAST.py`）

继承 `TestNet` 的设计模式，将 `TemporalSmoother` 集成进前向推理：

```python
class VideoTestNet(nn.Module):
    def forward(self, content, style, alpha=1.0):
        # 1. 提取 + 平滑风格调制信号
        style_feats = self.style_encoder(style)
        style_feats = self.smoother.smooth_style(style_feats)

        w, b = self.modulator(style_feats)
        w, b = self.smoother.smooth_modulation(w, b)

        # 2. 提取 + 平滑内容特征 ← 核心步骤
        content_feats = self.content_encoder(content)
        content_feats = self.smoother.smooth_content(content_feats)

        # 3. 用平滑后的信号解码
        return self.decoder(content_feats, style_feats, w, b, alpha)
```

### 3.3 视频推理脚本（`test_video_microAST.py`）

新脚本，支持：

- 逐帧处理目录中的图像（按文件名排序确保时序正确）
- `--temporal_momentum` 控制平滑强度
- `--scene_cut_frames` 手动标记场景切换帧（避免跨场景信号泄漏）
- Style image 缓存：相同 style 不重复编码
- 帧耗时统计

**使用示例**：

```bash
# 基础视频风格迁移
python test_video_microAST.py \
    --content_dir path/to/video_frames/ \
    --style path/to/style.jpg \
    --temporal_momentum 0.7

# 带场景切换标记
python test_video_microAST.py \
    --content_dir path/to/video_frames/ \
    --style path/to/style.jpg \
    --temporal_momentum 0.8 \
    --scene_cut_frames "0,150,300,450"
```

## 4. Part 2: 光流时序一致性损失（训练时）

### 4.1 `TemporalConsistencyLoss` 类（`temporal_loss.py`）

在训练阶段引入时序一致性约束，从网络权重层面学习产生时序稳定的特征。

**工作原理**：

```
             Frame t ──→ MicroAST ──→ Stylized_t ──┐
                                                    ├→ Warp(Stylized_t, flow) ≈ Stylized_t+1 ?
             Frame t+1 → MicroAST ──→ Stylized_t+1 ┘

             惩罚：|Warp(Stylized_t) - Stylized_t+1| × occlusion_mask
```

**遮挡检测**：当同时提供正向和反向光流时，通过 forward-backward consistency check 自动检测遮挡区域，排除在损失计算之外（遮挡区域的内容确实发生了变化，不应惩罚）。

**核心实现**：

```python
class TemporalConsistencyLoss(nn.Module):
    def forward(self, output_t, output_t1, flow_t_to_t1, flow_t1_to_t=None):
        # 1. Warp output_t 对齐到 output_t+1 的坐标
        warped = backward_warp(output_t, flow_t_to_t1)

        # 2. 计算逐像素差异（L1 或 L2）
        diff = |warped - output_t1|  # or (warped - output_t1)^2

        # 3. 遮挡掩码：排除不可靠区域
        mask = occlusion_mask(flow_t_to_t1, flow_t1_to_t)

        # 4. 仅在非遮挡区域计算损失
        loss = (diff * mask).sum() / mask.sum()
        return loss
```

**辅助函数**：

| 函数 | 作用 |
|------|------|
| `backward_warp(image, flow)` | 用反向光流将图像 warping 到目标坐标 |
| `flow_consistency_mask(fwd, bwd)` | Forward-backward check 生成遮挡掩码 |

### 4.2 训练集成方式

光流损失需要视频片段数据，集成到训练循环需要以下改动（本仓库提供 loss 函数，训练循环改动由用户按需完成）：

```python
# 概念示例（未修改 train_microAST.py）
temp_loss_fn = TemporalConsistencyLoss(loss_type='l1')

for batch in dataloader:
    content_t, content_t1, flow, flow_inv = batch  # 相邻帧对 + 光流

    output_t = model(content_t, style)
    output_t1 = model(content_t1, style)

    loss_temp = temp_loss_fn(output_t, output_t1, flow, flow_inv)
    total_loss = λ_c * loss_content + λ_s * loss_style
               + λ_ssc * loss_contrastive
               + λ_temp * loss_temp  # 新增时序损失
```

光流可通过以下预训练模型离线预计算：
- [RAFT](https://github.com/princeton-vl/RAFT)（高精度，速度较慢）
- [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch)（速度快）
- [PWC-Net](https://github.com/NVlabs/PWC-Net)（精度与速度平衡）

## 5. 新增/修改文件清单

### 新增文件

| 文件 | 内容 |
|------|------|
| `temporal_smoother.py` | `TemporalSmoother` 类 — EMA 时序平滑 |
| `temporal_loss.py` | `TemporalConsistencyLoss` 类 + `backward_warp` + `flow_consistency_mask` |
| `test_video_microAST.py` | 视频推理脚本（支持场景切换标记） |

### 修改文件

| 文件 | 改动 |
|------|------|
| `net_microAST.py` | 新增 `VideoTestNet` 类（wrap TestNet + TemporalSmoother） |

## 6. 参数增量

| 组件 | 参数量 |
|------|--------|
| `TemporalSmoother` | **0**（纯 EMA 运算，无学习参数） |
| `TemporalConsistencyLoss` | **0**（仅计算 loss，不属于模型） |
| `VideoTestNet` | 使用与 TestNet 完全相同的 Encoder/Decoder/Modulator，0 增量 |
| **总计** | **0** |

## 7. 累计改进总结

三次算法改进的总参数增量：

| 改进 | 参数增量 | 维度 |
|------|----------|------|
| SE 通道注意力 | +8,192 | 通道 — 哪些通道更重要 |
| 空间结构感知 | +9,378 | 空间 — 哪些区域保留结构 |
| 时序一致性（本次） | **0** | 时间 — 帧间特征稳定 |
| **合计** | **+17,570** | |
