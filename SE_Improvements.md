# SE (Squeeze-and-Excitation) 通道注意力改进

## 1. 问题背景：Representation Bottleneck（表征瓶颈）

MicroAST 为实现极速推理，采用了极浅的微型编码器（仅有一个下采样阶段 + 一个残差块），深度远低于 VGG-19 等传统骨干网络。这带来了一个物理上的硬限制：

- **浅层编码器只能提取浅层特征**：颜色分布、简单纹理统计信息
- **无法捕获复杂的艺术笔触组合**：面对构图复杂、包含多种细腻笔触的艺术品，微型编码器会将这些复杂特征“揉碎”并混淆
- **风格信号高度同质化**：编码器对所有风格图片输出“差不多”的特征表示，导致风格迁移结果缺乏多样性

## 2. 改进思路

在不打破 MicroAST“极速”约束的前提下，引入 **Squeeze-and-Excitation (SE) 轻量级通道注意力机制**。

SE 模块的核心思想：

```
输入 feature map [B, C, H, W]
         │
         ▼
  Squeeze: Global Average Pooling → [B, C, 1, 1]
         │
         ▼
  Excitation: FC(C) → ReLU → FC(C//r) → FC(r×C) → FC(C) → Sigmoid
         │
         ▼
  Scale: 将权重广播乘回原始 feature map
         │
         ▼
输出 feature map [B, C, H, W] （通道维度被重新加权）
```

- **Squeeze**：将每个通道的全局空间信息压缩为一个标量
- **Excitation**：通过两层全连接网络学习通道之间的依赖关系，产生 0~1 之间的注意力权重
- **Scale**：用注意力权重重新调整各通道的重要性

**为什么 SE 适合 MicroAST**：
- 计算开销极小（< 1% FLOPs 增量）
- 参数增量可忽略（每个 SE 块仅约 2K 参数）
- 不改变 feature map 的空间尺寸，可即插即用
- 赋予网络“权重分配”能力：自主强化代表性特征通道，抑制无效噪声

## 3. 具体修改

### 修改文件

`net_microAST.py`

### 3.1 新增 `SELayer` 类

```python
class SELayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)      # Squeeze
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # C → C/r
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),  # C/r → C
            nn.Sigmoid()                                  # 权重归一化到 [0,1]
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Squeeze
        y = self.fc(y).view(b, c, 1, 1)  # Excitation
        return x * y.expand_as(x)        # Scale
```

- **reduction=4**：对于 64 通道的特征图，瓶颈维度为 64/4 = 16，参数增量约 64×16 + 16×64 = 2048

### 3.2 `Encoder` 改动

在每个阶段输出后插入 SE 块：

```
Before:  x → enc1 → [x1] → enc2 → [x2]

After:   x → enc1 → SE1 → [x1]   → enc2 → SE2 → [x2]
                   精炼浅层特征              精炼深层特征
```

代码变更：

```python
# __init__ 新增
self.se1 = SELayer(int(64*slim_factor), reduction=4)
self.se2 = SELayer(int(64*slim_factor), reduction=4)

# forward 新增
x1 = self.se1(x1)   # SE 精炼 enc1 的输出
x2 = self.se2(x2)   # SE 精炼 enc2 的输出
```

> 因为 content encoder 和 style encoder 共享 `Encoder` 类，**两个编码器都会获得通道注意力增强**。这同时提升了内容结构提取和风格特征提取的质量。

### 3.3 `Decoder` 改动

在两个残差块输出后插入 SE 块：

```
Before:
  x1 → dec1 → x2 → featMod → x3 → dec2 → x4 → dec3 → out

After:
  x1 → dec1 → SE1 → x2 → featMod → x3 → dec2 → SE2 → x4 → dec3 → out
           精炼第一层             精炼第二层
```

代码变更：

```python
# __init__ 新增
self.se1 = SELayer(int(64*slim_factor), reduction=4)
self.se2 = SELayer(int(64*slim_factor), reduction=4)

# forward 新增
x2 = self.se1(x2)   # SE 精炼 dec1 的输出
x4 = self.se2(x4)   # SE 精炼 dec2 的输出
```

> dec3（上采样 + 输出 3 通道 RGB）不加 SE，因为其输出通道数为 3，不适合做通道注意力。

### 3.4 未改动的部分

| 模块 | 不改动原因 |
|------|-----------|
| `Modulator` | 已经是微型网络，输出 1×1 调制信号，SE 在此无意义 |
| `ResidualLayer` | 保持其通用性，SE 放在更高层次更有效 |
| `Net` / `TestNet` | 组合 Encoder/Decoder/Modulator，自动继承改进 |
| `vgg` | 仅用于训练时的 loss 计算，冻结参数 |

## 4. 参数增量分析

以 `slim_factor=1`（默认 64 通道）为例：

| 位置 | SE 块 | 参数增量 |
|------|-------|----------|
| Encoder.se1 | 64→16→64 | 64×16 + 16×64 = **2,048** |
| Encoder.se2 | 64→16→64 | 64×16 + 16×64 = **2,048** |
| Decoder.se1 | 64→16→64 | 64×16 + 16×64 = **2,048** |
| Decoder.se2 | 64→16→64 | 64×16 + 16×64 = **2,048** |
| **合计** | | **8,192 参数** |

- 仅增加约 8K 参数，而原模型参数量约为数十万级别
- SE 的 `AdaptiveAvgPool2d` + 两个小 `Linear` 层的 FLOPs 可忽略不计（< 1%）
- 推理速度几乎不受影响

## 5. 预期效果

1. **更丰富的风格表达**：通道注意力让编码器学会对不同的艺术风格激活不同的特征通道，缓解风格信号同质化问题
2. **更强的细节保持**：解码器中的 SE 帮助在上采样重建过程中聚焦关键特征，减少细节丢失
3. **几乎零速度代价**：SE 是纯 lightweight 操作，不改变空间分辨率，保持 MicroAST 的“极速”特性

## 6. 向后兼容性说明

旧 checkpoint（`models/*.pth.tar`）的 state_dict key 与修改后的网络不匹配，无法直接加载。

| 网络 | 新增 key | 无法匹配的旧 key |
|------|---------|-----------------|
| Encoder | `se1.fc.0.weight`, `se1.fc.0.bias`, `se1.fc.2.weight`, `se1.fc.2.bias`, `se2.*` (4 keys) | 无 |
| Decoder | 同上 8 keys | 无 |

**需要重新训练**以获得 SE 增强后的模型权重。
