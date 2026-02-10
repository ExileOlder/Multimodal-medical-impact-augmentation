# Setup Instructions

本文档提供详细的安装和配置指南。关于服务器环境的快速配置，请参考 [SERVER_SETUP.md](SERVER_SETUP.md)。

## 目录

- [依赖安装](#依赖安装)
- [依赖检查](#依赖检查)
- [配置文件](#配置文件)
- [测试策略](#测试策略)
- [数据集准备](#数据集准备)
- [A100 服务器优化](#a100-服务器优化)

## 依赖安装

### 核心原则：手动安装而非自动化脚本

本项目采用**手动依赖管理**的方式，避免自动化脚本可能引入的版本冲突和不必要的包。

### 为什么手动安装？

1. **避免版本冲突**：自动化脚本可能覆盖服务器上已有的包
2. **利用已安装的包**：服务器上可能已经安装了 PyTorch、CUDA 等大型包
3. **更好的控制**：明确知道安装了哪些包及其版本
4. **易于排查问题**：出现问题时更容易定位原因

### 安装步骤

#### 1. 检查已安装的包

在安装任何依赖之前，先检查服务器上已有的包：

```bash
python scripts/check_dependencies.py
```

该脚本会输出：
- 已安装的关键包及其版本
- 缺失的包列表
- PyTorch 和 CUDA 版本兼容性检查
- 相关警告信息

示例输出：
```
=== 依赖检查报告 ===
已安装包数量: 156
✓ torch: 2.0.1
✓ torchvision: 0.15.2
✓ numpy: 1.24.3
✓ pillow: 10.0.0
✗ gradio: 未安装
✓ PyTorch 和 CUDA 版本兼容

缺失的包:
- gradio
- jsonlines
```

#### 2. 安装常规依赖

根据依赖检查报告，手动安装缺失的包：

```bash
# 方式 1：安装所有依赖（如果大部分包都缺失）
pip install -r requirements.txt

# 方式 2：仅安装缺失的包（推荐）
pip install gradio jsonlines
```

**注意**：
- 如果某些包已经安装且版本合适，可以跳过
- 如果版本不匹配，根据需要升级或降级

#### 3. 手动安装 flash-attn（可选但推荐）

flash-attn 需要编译，单独安装：

```bash
# 检查 CUDA 版本
nvcc --version

# 安装 flash-attn（需要 5-10 分钟）
pip install flash-attn --no-build-isolation
```

**如果安装失败**：
1. 检查 CUDA 版本是否 >= 11.8
2. 检查 gcc 版本是否 >= 7.0
3. 如果无法解决，可以跳过（性能会降低 20-30%）

#### 4. 验证安装

运行冒烟测试验证安装：

```bash
python scripts/smoke_test.py
```

如果所有测试通过，说明依赖安装成功。

### 依赖列表说明

`requirements.txt` 中的依赖分为以下几类：

```txt
# 核心深度学习框架
torch>=2.0.0
torchvision>=0.15.0

# 图像处理
Pillow>=10.0.0
opencv-python>=4.8.0

# 数值计算
numpy>=1.24.0
scipy>=1.10.0

# 数据处理
PyYAML>=6.0
jsonlines>=3.1.0

# Web 界面
gradio>=4.0.0

# 评估指标
scikit-image>=0.21.0
lpips>=0.1.4

# 工具
tqdm>=4.65.0
tensorboard>=2.13.0

# 测试（可选）
pytest>=7.4.0

# 注意：flash-attn 需要手动安装
# pip install flash-attn --no-build-isolation
```

### 常见问题

#### Q: 版本冲突怎么办？

A: 手动卸载冲突的包，然后重新安装：
```bash
pip uninstall <conflicting_package>
pip install <package>==<version>
```

#### Q: 是否需要虚拟环境？

A: 推荐使用虚拟环境，但不是必须的：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

#### Q: 如何更新依赖？

A: 先检查当前版本，再决定是否更新：
```bash
python scripts/check_dependencies.py
pip install --upgrade <package>
```

## 依赖检查

### 使用 check_dependencies.py

`check_dependencies.py` 脚本提供全面的依赖检查功能。

### 功能

1. **列出已安装的包**：显示所有已安装的 Python 包及其版本
2. **检查关键依赖**：验证核心包（torch、numpy、pillow 等）是否安装
3. **CUDA 兼容性检查**：验证 PyTorch 和 CUDA 版本是否兼容
4. **缺失包检测**：对比 requirements.txt，找出缺失的包
5. **版本警告**：标记可能的版本冲突

### 使用方法

```bash
# 基本用法
python scripts/check_dependencies.py

# 查看详细输出
python scripts/check_dependencies.py --verbose
```

### 输出解读

```
=== 依赖检查报告 ===
已安装包数量: 156

关键依赖检查:
✓ torch: 2.0.1
✓ torchvision: 0.15.2
✓ numpy: 1.24.3
✓ pillow: 10.0.0
✗ gradio: 未安装

CUDA 兼容性:
✓ PyTorch 和 CUDA 版本兼容
  PyTorch CUDA: 11.8
  系统 CUDA: 11.8

缺失的包:
- gradio
- jsonlines

警告:
⚠️ flash-attn 需要手动编译安装
```

### 何时运行依赖检查

- ✅ 首次安装依赖前
- ✅ 安装依赖后验证
- ✅ 遇到导入错误时
- ✅ 更新依赖后
- ✅ 切换服务器环境时

## 配置文件

### 路径适配

所有配置文件中的路径必须符合以下规则：

1. **优先使用相对路径**
2. **必要时使用服务器限制路径**（`/home/Backup/maziheng`）
3. **避免硬编码绝对路径**

### 检查路径配置

使用 `check_paths.py` 脚本检查配置文件中的路径：

```bash
python scripts/check_paths.py
```

示例输出：
```
=== 路径检查报告 ===
检查文件: configs/train_config.yaml
✓ data.train_data: "data/train.jsonl" (相对路径)
✓ data.val_data: "data/val.jsonl" (相对路径)
✓ output.checkpoint_dir: "./checkpoints" (相对路径)
✓ output.log_dir: "./logs" (相对路径)

检查文件: configs/inference_config.yaml
✓ 所有路径检查通过

总结: 所有路径配置正确
```

### 路径配置示例

#### 正确的配置

```yaml
# configs/train_config.yaml
data:
  train_data: "data/train.jsonl"  # ✅ 相对路径
  val_data: "data/val.jsonl"      # ✅ 相对路径

output:
  checkpoint_dir: "./checkpoints"  # ✅ 相对路径
  log_dir: "./logs"                # ✅ 相对路径
```

#### 错误的配置

```yaml
# ❌ 硬编码绝对路径
data:
  train_data: "/home/user/data/train.jsonl"

# ❌ 使用用户主目录
data:
  train_data: "~/data/train.jsonl"
```

#### 必要时使用绝对路径

如果必须使用绝对路径，确保使用服务器限制路径：

```yaml
# ✅ 使用服务器限制路径
data:
  train_data: "/home/Backup/maziheng/medical-image-augmentation-system/data/train.jsonl"
```

### A100 优化配置

配置文件已针对 A100 GPU 进行优化。

#### 批次大小配置

```yaml
training:
  batch_size: 2  # 针对 1024×1024 图像
  gradient_accumulation_steps: 16  # 确保 Effective Batch Size = 32
```

**Effective Batch Size 计算**：
```
Effective Batch Size = batch_size × gradient_accumulation_steps
                     = 2 × 16
                     = 32
```

**建议**：
- 对于 1024×1024 图像：Effective Batch Size >= 32
- 对于 512×512 图像：Effective Batch Size >= 64
- 如果显存不足，减小 batch_size 并增加 gradient_accumulation_steps

#### 混合精度配置

```yaml
training:
  mixed_precision: "bf16"  # A100 优化
```

**说明**：
- A100 支持 BF16（Brain Float 16），比 FP16 更稳定
- BF16 不需要 loss scaling，训练更简单
- 如果使用其他 GPU（如 V100），改为 "fp16"

#### 数据加载配置

```yaml
training:
  num_workers: 8  # 充分利用 CPU
```

**建议**：
- A100 服务器通常有多核 CPU，设置 num_workers = 4-8
- 如果数据加载成为瓶颈，可以增加到 16
- 如果 CPU 资源有限，减小到 4

#### 验证配置

使用 `validate_config.py` 验证配置的合理性：

```bash
python scripts/validate_config.py
```

示例输出：
```
=== 配置验证报告 ===
✓ 批次配置: Effective Batch Size: 32 (合理)
估算显存需求: 28.45 GB
✓ 显存需求在 A100 40GB 范围内

建议:
- 当前配置适合 A100 40GB
- 如果使用 A100 80GB，可以增加 batch_size 到 4
```

## 测试策略

### 简化的测试方法

本项目采用**简化的测试策略**，避免过度工程化：

1. **冒烟测试**（Smoke Test）：快速验证核心功能
2. **单元测试**（Unit Test）：测试关键函数

**不包含**：
- ❌ 属性测试（Hypothesis）：对单人毕设项目过于复杂
- ❌ 集成测试：在快速迭代阶段不必要
- ❌ 性能测试：A100 性能足够

### 冒烟测试

#### 目的

快速验证核心功能是否正常，包括：
- 数据加载
- 模型初始化
- 前向传播

#### 使用方法

```bash
python scripts/smoke_test.py
```

#### 执行时间

< 2 分钟

#### 示例输出

```
=== 冒烟测试开始 ===

测试数据加载...
✓ 数据加载测试通过

测试模型初始化...
✓ 模型初始化测试通过

测试前向传播...
✓ 前向传播测试通过

=== 冒烟测试结果 ===
通过: 3/3
✓ 所有测试通过，可以安全删除 codes_backup/
```

#### 何时运行冒烟测试

- ✅ 安装依赖后
- ✅ 清理项目结构前后
- ✅ 修改核心代码后
- ✅ 切换服务器环境后

### 单元测试

#### 目的

测试关键函数的正确性，包括：
- 数据预处理
- 模型组件

#### 使用方法

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python tests/test_preprocessing.py
python tests/test_model.py
```

#### 测试文件

1. **tests/test_preprocessing.py**：
   - 图像调整大小
   - 掩码调整大小
   - 像素值归一化

2. **tests/test_model.py**：
   - 模型初始化
   - 前向传播
   - 输出形状验证

#### 示例输出

```
============================= test session starts ==============================
collected 5 items

tests/test_preprocessing.py ...                                          [ 60%]
tests/test_model.py ..                                                   [100%]

============================== 5 passed in 2.34s ===============================
```

### 测试覆盖范围

本项目的测试策略专注于：
- ✅ 核心功能验证
- ✅ 关键函数测试
- ✅ 快速执行（< 5 分钟）

不追求：
- ❌ 100% 代码覆盖率
- ❌ 边界情况的穷举测试
- ❌ 复杂的测试框架

### 为什么简化测试？

1. **单人开发**：不需要复杂的测试流程
2. **快速迭代**：测试不应成为开发瓶颈
3. **实用导向**：专注核心功能，不过度测试
4. **毕设项目**：时间有限，优先实现功能

## 数据集准备

## 数据集准备

### 推荐数据集：FGADR

1. **下载数据集**：https://github.com/csyizhou/FGADR-2842-Dataset
2. **解压到**：`data/FGADR/`
3. **创建 JSONL 文件**（见下文）

### JSONL 格式

创建 `data/train/data.jsonl`：

```jsonl
{"image_path": "data/FGADR/images/001.png", "caption": "2", "mask_path": "data/FGADR/masks/001.png"}
{"image_path": "data/FGADR/images/002.png", "caption": "Mild diabetic retinopathy", "mask_path": "data/FGADR/masks/002.png"}
{"image_path": "data/FGADR/images/003.png", "caption": "3", "mask_path": null}
```

字段说明：
- `image_path`: 眼底图像路径
- `caption`: DR 分级（0-4）或文本描述
- `mask_path`: 分割掩码路径（可选，null 表示仅文本模式）

### 验证数据加载

```bash
python scripts/smoke_test.py
```

## A100 服务器优化

### 硬件规格

- **GPU**：NVIDIA A100 40GB / 80GB
- **CUDA**：11.8+
- **显存**：40GB / 80GB

### 优化配置

#### 1. Flash Attention 2

Flash Attention 2 可提供 2-3x 训练加速：

```bash
pip install flash-attn --no-build-isolation
```

**性能对比**：
- 无 Flash Attention：~1.2 it/s
- 有 Flash Attention：~3.5 it/s

#### 2. BF16 混合精度

A100 支持 BF16，比 FP16 更稳定：

```yaml
# configs/train_config.yaml
training:
  mixed_precision: "bf16"
```

**优势**：
- 不需要 loss scaling
- 数值稳定性更好
- 训练速度提升 ~2x

#### 3. 批次大小

根据显存调整批次大小：

```yaml
# A100 40GB
training:
  batch_size: 2
  gradient_accumulation_steps: 16  # Effective Batch Size = 32

# A100 80GB
training:
  batch_size: 4
  gradient_accumulation_steps: 8  # Effective Batch Size = 32
```

#### 4. 数据加载

充分利用 CPU 资源：

```yaml
training:
  num_workers: 8
  prefetch_factor: 2
  persistent_workers: true
```

### 性能基准

在 A100 40GB 上的预期性能：

| 配置 | 批次大小 | 显存使用 | 训练速度 |
|------|---------|---------|---------|
| 1024×1024, BF16, Flash Attn | 2 | ~28 GB | ~3.5 it/s |
| 512×512, BF16, Flash Attn | 8 | ~32 GB | ~8.0 it/s |

### 故障排查

#### 显存不足（OOM）

1. 减小 batch_size
2. 减小 image_size
3. 启用 gradient_checkpointing

#### 训练速度慢

1. 确保安装了 flash-attn
2. 使用 BF16 混合精度
3. 增加 num_workers

## 下一步

安装完成后：

1. ✅ 运行依赖检查：`python scripts/check_dependencies.py`
2. ✅ 运行路径检查：`python scripts/check_paths.py`
3. ✅ 验证配置：`python scripts/validate_config.py`
4. ✅ 运行冒烟测试：`python scripts/smoke_test.py`
5. ✅ 准备数据集
6. ✅ 开始训练：`python train.py --config configs/train_config.yaml`

## 参考文档

- **快速启动**：[README.md](README.md)
- **服务器配置**：[SERVER_SETUP.md](SERVER_SETUP.md)
- **清理检查清单**：[CLEANUP_CHECKLIST.md](CLEANUP_CHECKLIST.md)（如果需要清理项目结构）

## 常见问题

### Q: 如何选择合适的批次大小？

A: 使用 `validate_config.py` 估算显存需求：
```bash
python scripts/validate_config.py
```

### Q: 是否必须使用 flash-attn？

A: 不是必须的，但强烈推荐。没有 flash-attn 性能会降低 20-30%。

### Q: 如何验证配置是否正确？

A: 运行以下脚本：
```bash
python scripts/check_paths.py
python scripts/validate_config.py
python scripts/smoke_test.py
```

### Q: 训练不稳定怎么办？

A: 检查 Effective Batch Size 是否 >= 32：
```bash
python scripts/validate_config.py
```

## 联系支持

如有问题，请：
1. 查看 [SERVER_SETUP.md](SERVER_SETUP.md) 的常见问题章节
2. 运行诊断脚本收集信息
3. 查看日志文件：`logs/cleanup.log`, `logs/train.log`
