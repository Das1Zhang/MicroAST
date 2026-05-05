# MicroAST 改进版训练指南 (Colab)

## 0. 前置说明

`train_microAST.py` 本身**无需修改**。训练循环通过 `net.Encoder()` / `net.Decoder()` 实例化模型，
SE 通道注意力和空间结构感知分支已内置在这些类中，训练时会自动参与前向和反向传播。

三次算法改进的参数量：

| 改进组件 | 所属模块 | 参数增量 |
|----------|----------|----------|
| SELayer ×4 | Encoder ×2 + Decoder ×2 | +8,192 |
| StructureAwarenessBranch ×2 | Decoder ×2 | +9,378 |
| TemporalSmoother | VideoTestNet（推理专用） | 0 |
| **合计** | | **+17,570** |

与原始 MicroAST（不含 VGG）参数量相比，增幅约 3-5%，训练速度和显存占用几乎不受影响。

---

## 1. Colab 环境配置

### 1.1 上传代码文件

将以下文件上传到 Colab 工作目录（`/content/`）：

```
# 必须
net_microAST.py         # 模型定义（含 SE + StructureAwarenessBranch + VideoTestNet）
train_microAST.py       # 训练脚本（未修改，可直接使用）
function.py             # AdaIN / calc_mean_std
sampler.py              # 无限采样器

# VideoTestNet 懒加载需要（训练时不使用，但 import net_microAST 时需要文件存在）
temporal_smoother.py    # EMA 时序平滑器
temporal_loss.py        # 光流时序损失（训练时不需要，可选）
```

### 1.2 安装依赖

```python
# Colab Cell
!pip install tensorboardX tqdm opencv-python scikit-image thop
```

> PyTorch 和 torchvision 在 Colab 中已预装，无需额外安装。
>
> 如果 Colab 中 PyTorch 版本较新（≥2.0），需要确认 CUDA 兼容性：
> ```python
> import torch; print(torch.__version__, torch.cuda.is_available())
> ```

### 1.3 准备 VGG 预训练权重

VGG 权重用于训练时计算 content loss 和 style loss（**推理时不需要**）。

```python
# Colab Cell — 从 Google Drive 下载
!gdown "https://drive.google.com/uc?id=1PUXro9eqHpPs_JwmVe47xY692N3-G9MD" -O models/vgg_normalised.pth
```

确认文件就位：
```python
!ls -lh models/vgg_normalised.pth
# 预期大小约 254MB
```

---

## 2. 数据集准备

### 2.1 MS-COCO（内容图片）

```bash
# 下载 2017 Train Images（约 18GB，118K 张图片）
!wget http://images.cocodataset.org/zips/train2017.zip
!unzip -q train2017.zip -d coco2014/
!mv coco2014/train2017 coco2014/train2014  # 对齐默认路径
```

> 如果磁盘空间有限，也可以用较小数据集替代（如 COCO val2017，5K 张）。
> 修改训练命令中的 `--content_dir` 即可。

### 2.2 WikiArt（风格图片）

WikiArt 数据集需从 Kaggle 下载，需要 Kaggle API：

```python
# Colab Cell
!pip install kaggle
# 上传 kaggle.json 或通过环境变量配置
!mkdir -p ~/.kaggle
# 将 kaggle.json 放到 ~/.kaggle/ 下，然后：
!kaggle datasets download -d steubk/wikiart -p wikiart/
!unzip -q wikiart/wikiart.zip -d wikiart/
```

> WikiArt 包含约 80K 张绘画作品。
>
> 也可使用更小的风格数据集（如 [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers)）替代。

### 2.3 Google Drive 挂载（必需）

Colab `/content/` 中的文件在断连或运行时结束后**不会持久保存**。数据集和 checkpoint 必须放在 Google Drive 中：

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 3. 开始训练

### 3.1 基础训练命令

与原始 MicroAST 一致，只需指定数据集路径，所有参数均有默认值：

```bash
!python train_microAST.py \
    --content_dir ./coco2014/train2014 \
    --style_dir ./wikiart/train
```

### 3.2 Checkpoint 持久化（重要）

默认 `--save_dir ./exp` 和 `--checkpoints ./checkpoints` 保存在 `/content/` 下，断连即丢失。**必须指向 Google Drive 路径**：

```bash
!python train_microAST.py \
    --content_dir ./coco2014/train2014 \
    --style_dir ./wikiart/train \
    --save_dir /content/drive/MyDrive/MicroAST/exp \
    --checkpoints /content/drive/MyDrive/MicroAST/checkpoints \
    --log_dir /content/drive/MyDrive/MicroAST/logs \
    --sample_path /content/drive/MyDrive/MicroAST/samples
```

所有训练产物说明：

| 参数 | 默认路径 | 内容 | 持久需求 |
|------|----------|------|----------|
| `--save_dir` | `./exp` | Content/Style Encoder、Decoder、Modulator 的 `.pth.tar` | **必须指向 Drive** |
| `--checkpoints` | `./checkpoints` | 完整 checkpoint（含 optimizer，用于 `--resume`） | **必须指向 Drive** |
| `--log_dir` | `./logs` | TensorBoard 日志 | 可选 |
| `--sample_path` | `./samples` | 中间采样图（每 500 步） | 可选 |

### 3.2 常用可选参数

仅在需要调整时添加，其余参数使用脚本内置默认值即可：

| 参数 | 默认值 | 何时调整 |
|------|--------|----------|
| `--batch_size` | 8 | **Colab T4 上建议保持默认**。OOM 时改为 4 |
| `--n_threads` | 16 | **Colab 建议改为 2-4**，避免多进程卡死 |
| `--max_iter` | 160000 | 快速验证时临时设为 5000 或 10000 |
| `--gpu_id` | 0 | 单 GPU 无需改动 |

### 3.3 训练产出

训练过程中会在以下目录产生文件：

```
exp/
├── content_encoder_iter_10000.pth.tar    # Content Encoder 权重
├── style_encoder_iter_10000.pth.tar      # Style Encoder 权重
├── modulator_iter_10000.pth.tar          # Modulator 权重
├── decoder_iter_10000.pth.tar            # Decoder 权重（含 SE + StructureBranch）
├── ...（每 10K 步一组）
├── content_encoder_iter_160000.pth.tar   # 最终
├── style_encoder_iter_160000.pth.tar
├── modulator_iter_160000.pth.tar
└── decoder_iter_160000.pth.tar

checkpoints/
└── checkpoints.pth.tar                   # 完整 checkpoint（含 optimizer state，用于 resume）

samples/
└── output000500.jpg                      # 每 500 步的中间采样（content + style + result 拼接）

logs/
└── events.out.tfevents.*                 # TensorBoard 日志
```

### 3.4 断点续训

```bash
!python train_microAST.py \
    --content_dir ./coco2014/train2014 \
    --style_dir ./wikiart/train \
    --resume \
    --checkpoints /content/drive/MyDrive/MicroAST/checkpoints \
    --save_dir /content/drive/MyDrive/MicroAST/exp
```

恢复训练时会加载 `checkpoints/checkpoints.pth.tar` 中的模型权重、optimizer 状态和 epoch 计数。

---

## 4. 训练后验证

### 4.1 单图测试

将最终 checkpoint 复制到 `models/` 目录（或直接修改路径）：

```python
import shutil, glob, os

# 复制最终 checkpoint
for f in glob.glob('exp/*_iter_160000.pth.tar'):
    shutil.copy(f, 'models/')

# 测试
!python test_microAST.py \
    --content inputs/content/1.jpg \
    --style inputs/style/1.jpg \
    --output output_test
```

### 4.2 视频测试

```bash
!python test_video_microAST.py \
    --content_dir path/to/video_frames/ \
    --style path/to/style.jpg \
    --temporal_momentum 0.7 \
    --output output_video
```

### 4.3 观察训练过程

```python
# Colab Cell — 启动 TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs/
```

---

## 5. 光流时序损失的集成（可选，进阶）

光流损失 (`temporal_loss.py`) 需要视频片段数据和预计算的光流。
以下是在训练脚本中集成的概念示例（需自行修改训练循环）：

```python
# === 附加到 train_microAST.py 中的训练循环 ===
from temporal_loss import TemporalConsistencyLoss

temp_loss_fn = TemporalConsistencyLoss(loss_type='l1')

# 需要修改 DataLoader 以提供相邻帧对和光流
# content_pairs: [B, 2, 3, H, W] — 每对相邻帧
# flow_fwd, flow_bwd: [B, 2, H, W]

for i in range(start_iter+1, args.max_iter):
    content_pair, flow_fwd, flow_bwd = next(content_iter)
    content_t = content_pair[:, 0].to(device)
    content_t1 = content_pair[:, 1].to(device)
    flow_fwd = flow_fwd.to(device)
    flow_bwd = flow_bwd.to(device)

    output_t = network(content_t, style_images)
    output_t1 = network(content_t1, style_images)

    # 新增时序损失
    loss_temp = temp_loss_fn(output_t, output_t1, flow_fwd, flow_bwd)

    total_loss = loss_c + loss_s + loss_contrastive + 0.5 * loss_temp
    # ... backward, step ...
```

> 光流可通过 [RAFT](https://github.com/princeton-vl/RAFT) 离线预计算。
> 这部分改动较大，当前仓库中 `train_microAST.py` 未做修改，按需自行集成。

---

## 6. 常见问题

### OOM (Out of Memory)
- 降低 `--batch_size`（如 4 或 2）
- 降低 `--n_threads`
- 使用 Colab Pro+ 的 V100/A100

### DataLoader 卡死
- 将 `--n_threads` 设为 **0** 或 **2**
- 确认数据集路径正确、无损坏文件

### `module 'net_microAST' has no attribute 'xxx'`
- 确认上传的是改进后的 `net_microAST.py`（包含 SELayer、StructureAwarenessBranch）
- 确认 `temporal_smoother.py` 也在工作目录中

### 训练速度参考
- Colab T4 (16GB): batch_size=8, ~3-4 iterations/s
- 160K iterations 约需 12-15 小时

### old checkpoint 不兼容
- `models/` 中的旧 checkpoint 是原始 MicroAST 的，无法加载到改进后的网络
- 必须重新训练
