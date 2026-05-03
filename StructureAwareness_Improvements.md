# 空间结构感知改进 (Spatial Structure-Aware Modulation)

## 1. 问题背景：Spatial Misalignment（空间语义错乱）

MicroAST 的核心创新“双重调制（Dual-Modulation）”在本质上是**全局操作**：

```
FeatMod (AdaIN): 用 style 特征的全局均值和方差替换 content 特征的均值和方差
    ↓ 一张 style 图片 → 一组 (μ, σ) → 无差别作用于整张 content 图的所有空间位置

FilterMod: 用 style 信号生成全局卷积核权重，逐通道调制 decoder 的滤波器
    ↓ 同一组 filter weights/bias 作用于所有空间位置
```

**后果**：模型完全没有“空间感知力”——它不知道图片中哪部分是天空、哪部分是建筑、哪部分是人脸。最终风格迁移结果中，风格图像的草地笔触被生硬地贴到内容图像的人脸上，粗犷的线条破坏了本该平滑的背景，造成严重的**语义违和**。

这个问题的根因在于：**全局调制参数丢失了空间维度**。每个通道的 `(μ, σ)` 和 `(w, b)` 都是标量，无法表达“在 A 区域用风格 1，在 B 区域用风格 2”。

## 2. 改进思路

### 核心思想

在内容编码器提取特征的同时，**并联**一个极其轻量的空间结构感知分支（`StructureAwarenessBranch`），从内容特征中预测一张**结构保持图**（Structure Preservation Map），在注入风格调制信号时对每个像素进行加权分配：

```
全局 alpha:     所有位置统一使用同一个标量 alpha
           ↓
空间 alpha:     每个位置使用独立的 alpha 值（由结构图决定）
  - 高结构区域（边缘、纹理、轮廓）→ 降低风格化 → 保留内容
  - 低结构区域（平滑背景、天空） → 加强风格化
```

### 为什么放在 Decoder 内部

Decoder 的 `forward` 签名是 `forward(self, x, s, w, b, alpha)`，其中 `x` 就是内容编码器的特征。**Decoder 已经拿到了内容特征**，可以直接从中生成结构图，完全不需要修改 `Net`、`TestNet`、`Encoder` 等任何其他类。这是一种最低侵入性的设计。

### 为什么用两层结构图（深层 + 浅层）

Decoder 中有两次 AdaIN 注入：

| 注入点 | 输入特征 | 语义层级 |
|--------|----------|----------|
| 第一层 (x1) | `x[1]`（enc1 + enc2 输出） | 深层语义结构（物体边界、主体轮廓） |
| 第二层 (x3) | `x[0]`（enc1 输出） | 浅层纹理结构（局部边缘、细节） |

两层使用独立的结构分支，分别捕捉不同粒度的空间信息。

### 为什么使用 3×3 卷积而不是 1×1

1×1 卷积只能做逐像素的通道混合，无法感知空间邻域关系。而“结构”本质上是一种**空间模式**（例如边缘是相邻像素的突变），需要至少 3×3 的感受野才能检测。3×3 conv with padding=1 保持空间分辨率不变，参数量仅增加了 9 倍常数因子，依然在“极简”范畴内。

## 3. 具体修改

### 修改文件

`net_microAST.py`

### 3.1 新增 `StructureAwarenessBranch` 类

```python
class StructureAwarenessBranch(nn.Module):
    def __init__(self, channels):
        super(StructureAwarenessBranch, self).__init__()
        bottleneck = max(channels // 8, 4)  # 不低于 4 通道
        self.conv = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 3, padding=1),  # 3×3 感知邻域结构
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, 1, 3, padding=1),          # 输出单通道结构图
            nn.Sigmoid()  # 归一化到 [0, 1]
        )

    def forward(self, feat):
        return self.conv(feat)  # [B, C, H, W] → [B, 1, H, W]
```

参数分析（以 64 通道为例）：

| 层 | 参数量 |
|----|--------|
| Conv2d(64, 8, 3) | 64 × 8 × 3 × 3 + 8 = 4,616 |
| Conv2d(8, 1, 3) | 8 × 1 × 3 × 3 + 1 = 73 |
| **每分支合计** | **~4,689** |
| **两分支合计** | **~9,378** |

### 3.2 Decoder 新增成员变量

```python
self.structure_branch_1 = StructureAwarenessBranch(int(64*slim_factor))
self.structure_branch_2 = StructureAwarenessBranch(int(64*slim_factor))
```

### 3.3 Decoder.forward 改动

**Before（全局 alpha）：**

```python
x1 = featMod(x[1], s[1])
x1 = alpha * x1 + (1-alpha) * x[1]   # alpha 是标量
...
x3 = featMod(x2, s[0])
x3 = alpha * x3 + (1-alpha) * x2     # alpha 是标量
```

**After（空间感知 alpha）：**

```python
# 1. 从内容特征生成结构保持图
structure_map_1 = self.structure_branch_1(x[1])  # 深层结构
structure_map_2 = self.structure_branch_2(x[0])  # 浅层结构

# 2. 用结构图对 AdaIN 注入做逐像素加权
spatial_alpha_1 = alpha * (1 - structure_map_1)
x1 = spatial_alpha_1 * featMod(x[1], s[1]) + (1 - spatial_alpha_1) * x[1]
#   ↑ 结构强 → structure_map → 1 → spatial_alpha → 0 → 保留原始内容
#   ↑ 结构弱 → structure_map → 0 → spatial_alpha → alpha → 充分风格化

spatial_alpha_2 = alpha * (1 - structure_map_2)
x3 = spatial_alpha_2 * featMod(x2, s[0]) + (1 - spatial_alpha_2) * x2
```

### 3.4 未修改的部分

| 组件 | 不改原因 |
|------|----------|
| `Encoder` | 结构分支在 Decoder 中，不需要 Encoder 改动 |
| `Modulator` | 调制信号生成不受影响 |
| `Net` / `TestNet` | Decoder 接口不变，自动继承改进 |
| FilterMod（残差块内） | 保留全局调制，结构与风格的混合已在 AdaIN 层通过空间 alpha 完成 |

## 4. 与 SE 通道注意力的关系

两次改进是**正交且互补**的：

| 改进 | 维度 | 作用 |
|------|------|------|
| SE (Squeeze-Excitation) | **通道**维度 | 哪些通道更重要 → 强化代表性特征通道 |
| StructureAwarenessBranch | **空间**维度 | 哪些区域需要保留内容结构 → 逐像素控制风格强度 |

两次改进共增加约 17.4K 参数（SE: 8.2K + Structure: 9.4K），对推理速度的影响可忽略。

## 5. 预期效果

1. **语义保真**：人脸、文字、建筑等结构化区域的内容得到更好的保留
2. **自然过渡**：风格化区域与非风格化区域之间产生平滑的空间过渡，而非生硬的全局涂抹
3. **风格丰富度**：不同区域的特点（如边缘 vs 平坦区域）被差异化处理，风格表达更细腻
4. **零速度代价**：结构分支的 3×3 conv 运算量极低，不改变 MicroAST 的“极速”特性

## 6. 向后兼容性

旧 checkpoint 无法加载（Decoder 的 state_dict 新增了 `structure_branch_1.conv.0.weight` 等 8 个 key）。**需要重新训练**。
