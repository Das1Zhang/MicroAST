# MicroAST Code Review

## 整体评价

MicroAST 代码结构清晰，模型架构实现符合论文描述。以下是发现的问题，按严重程度排列。

---

## 严重问题

### 1. 强制 CUDA，无 CPU 回退

**文件：`test_microAST.py:64`，`metrics/calc_cs_loss.py:24`**

```python
# test_microAST.py:64
device = torch.device('cuda:%d' % args.gpu_id)  # 无 GPU 直接崩溃

# test_microAST.py:116-117 — CUDA-only 的同步调用也会崩溃
torch.cuda.synchronize()
```

没有 GPU 时会直接抛出 `RuntimeError`。建议改为：

```python
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
```

`metrics/calc_cs_loss.py:24` 同理，注释中虽有 CPU 方案但被注释掉了：

```python
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")  # 硬编码，无 GPU 时崩溃
```

---

### 2. 训练脚本缺少 `if __name__ == '__main__'` 保护

**文件：`train_microAST.py:90`**

`args = parser.parse_args()` 以及模型初始化、DataLoader 构建都在模块顶层执行。`import train_microAST` 时会立即触发参数解析和模型加载，这是 Python 的不良实践，也会导致多进程 DataLoader（`num_workers > 0`）在 Windows 上出现进程 fork 问题。

建议将训练逻辑包裹在：

```python
if __name__ == '__main__':
    args = parser.parse_args()
    # ... 其余代码
```

---

## 中等问题

### 3. 对比损失 O(B²) 嵌套 Python 循环

**文件：`net_microAST.py:275-298`**

```python
for i in range(int(style.size(0))):   # batch size
    for j in range(int(style.size(0))):  # batch size again
```

`batch_size=8` 时有 64 次迭代，全部在 Python 层面完成，无法利用 GPU 并行。此处可通过张量广播向量化，大幅降低训练时间，但不影响结果正确性。

---

### 4. `torch.svd` 已废弃

**文件：`function.py:45`**

```python
U, D, V = torch.svd(x)  # PyTorch 1.9+ 已废弃
```

应改为：

```python
U, D, Vh = torch.linalg.svd(x, full_matrices=False)
V = Vh.mH
```

> **注意**：`coral` 函数在主推理流程中**未被调用**（`net_microAST.py` 未导入它），不影响训练和推理运行，但若在 metrics 脚本中引入则会触发 DeprecationWarning。

---

### 5. 路径拼接未使用 `os.path.join`

**文件：`metrics/calc_cs_loss.py:135, 199`，`metrics/calc_ssim.py:28, 30`**

```python
cv2.imread(stylized_dir + stylized)      # 缺少路径分隔符
cv2.imread(content_dir + name[0] + '.jpg')
```

若调用时目录路径末尾没有 `/`，文件路径会拼接错误。应改为：

```python
cv2.imread(os.path.join(stylized_dir, stylized))
cv2.imread(os.path.join(content_dir, name[0] + '.jpg'))
```

---

## 轻微问题

### 6. `slim_factor` 硬编码为全局常量

**文件：`net_microAST.py:41`**

```python
slim_factor = 1  # 模块级全局变量
```

若想调整模型通道宽度，需要直接修改源码。更好的做法是将其作为 `Encoder`、`Decoder`、`Modulator` 的构造函数参数，以支持不同规模的模型配置。

---

### 7. `vgg` 在模块级实例化

**文件：`net_microAST.py:135`**

```python
vgg = nn.Sequential(...)  # import net_microAST 时立即创建
```

`import net_microAST` 时无论是否使用 VGG 都会分配该网络的内存。测试阶段使用 `TestNet`，完全不需要 VGG，却仍然承担了这部分开销。建议将 `vgg` 的实例化移至 `train_microAST.py` 内部。

---

## 问题汇总

| 编号 | 严重程度 | 文件 | 描述 |
|------|----------|------|------|
| 1 | 严重 | `test_microAST.py:64`, `metrics/calc_cs_loss.py:24` | 强制 CUDA，无 CPU 回退 |
| 2 | 严重 | `train_microAST.py:90` | 缺少 `if __name__ == '__main__'` 保护 |
| 3 | 中等 | `net_microAST.py:275-298` | 对比损失 O(B²) 嵌套 Python 循环 |
| 4 | 中等 | `function.py:45` | `torch.svd` 已废弃 |
| 5 | 中等 | `metrics/calc_cs_loss.py`, `metrics/calc_ssim.py` | 路径拼接未使用 `os.path.join` |
| 6 | 轻微 | `net_microAST.py:41` | `slim_factor` 硬编码为全局常量 |
| 7 | 轻微 | `net_microAST.py:135` | `vgg` 在模块级实例化 |
