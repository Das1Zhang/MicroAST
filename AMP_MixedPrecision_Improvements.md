# AMP 混合精度训练改进

## 1. 问题背景

原始 `train_microAST.py` 使用纯 FP32 训练，在 Colab T4（有 Tensor Core 但未被利用）上：
- GPU RAM 占用仅 3.3/15GB
- 预估训练时长达 30+ 小时
- 增大 batch_size 反而更慢（对比损失 O(B²) 膨胀）

## 2. 改进方案

启用 PyTorch **自动混合精度 (AMP)**，利用 T4 的 Tensor Core 在矩阵运算中自动使用 FP16，同时保持关键计算（如 loss）在 FP32 以防止精度损失。

### 工作原理

```
FP32 (原始):  所有操作都在 32-bit 浮点
    ↓ 显存占用大，Tensor Core 未利用

AMP (改进):
    autocast() → Conv/Linear/MatMul 自动用 FP16（Tensor Core 加速）
               → 数值敏感操作（softmax, loss, norm）保持 FP32
    GradScaler → 防止 FP16 梯度下溢：loss 放大 → backward → 缩小 → 更新
```

## 3. 具体修改

**文件：`train_microAST.py`**

### 3.1 新增导入（第 5 行）

```python
from torch.cuda.amp import autocast, GradScaler
```

### 3.2 创建 GradScaler（第 148 行）

```python
scaler = GradScaler()
```

### 3.3 前向计算包在 autocast 中（第 153-154 行）

```python
with autocast():
    stylized_results, loss_c, loss_s, loss_contrastive = network(content_images, style_images)
```

### 3.4 反向传播使用 scaler（第 161-163 行）

```python
optimizer.zero_grad()
scaler.scale(loss).backward()  # 放大 loss → 反向 → 缩小梯度
scaler.step(optimizer)         # 更新参数（自动 unscale）
scaler.update()                # 动态调整放大系数
```

### 3.5 断点续训支持

Checkpoint 保存/恢复加入 `scaler` 的 state_dict，确保 resume 后 AMP 状态一致。

## 4. 预期效果

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 训练速度 | 基准 | **1.5-2x** |
| GPU 利用率 | 低（Tensor Core 闲置） | 高（Tensor Core 主导计算） |
| 显存占用 | 约 3.3GB | 约 2-2.5GB（FP16 减半） |
| 训练结果质量 | — | 无差异 |

## 5. 使用方式

训练命令不变，AMP 自动生效：

```bash
python train_microAST.py \
    --content_dir ./coco2014/train2014 \
    --style_dir ./wikiart/train \
    --batch_size 8 \
    --n_threads 2
```

> 注意：batch_size 保持 8，因对比损失为 O(B²)，增大 batch 反而膨胀计算量。
