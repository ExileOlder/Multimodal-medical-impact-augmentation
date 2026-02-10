# 🏥 Medical Image Augmentation System

基于 RetinaLogos 的医学影像增广系统，支持文本和分割掩码条件的糖尿病视网膜病变（DR）眼底图像生成。

## ✨ 特性

- **多模态条件生成**：支持文本描述和分割掩码条件
- **Flow Matching 训练**：使用 Rectified Flow 进行高效训练
- **A100 优化**：Flash Attention 2 + BF16 混合精度训练
- **Gradio 演示界面**：简洁易用的 Web 界面
- **下游任务评估**：ResNet-50 分类实验验证增广价值

## 📋 目录

- [项目结构](#项目结构)
- [安装](#安装)
- [快速启动](#快速启动)
- [数据准备](#数据准备)
- [训练](#训练)
- [推理](#推理)
- [Gradio 演示](#gradio-演示)
- [下游评估](#下游评估)
- [简化说明](#简化说明)

## 📁 项目结构

简化后的项目结构如下：

```
medical-image-augmentation-system/
├── src/                    # 核心代码目录
│   ├── data/              # 数据加载和预处理
│   │   ├── jsonl_loader.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── models/            # 模型定义
│   │   ├── nexdit_mask.py
│   │   └── mask_utils.py
│   ├── training/          # 训练流程
│   │   ├── config.py
│   │   └── trainer.py
│   ├── inference/         # 推理和导出
│   │   ├── generator.py
│   │   └── export.py
│   ├── app/               # Gradio 应用
│   │   └── demo.py
│   └── evaluation/        # 评估工具
│       └── metrics.py
├── scripts/               # 工具脚本
│   ├── check_dependencies.py
│   ├── check_paths.py
│   ├── validate_config.py
│   ├── smoke_test.py
│   └── rollback.py
├── tests/                 # 测试文件
│   ├── test_preprocessing.py
│   └── test_model.py
├── configs/               # 配置文件
│   ├── train_config.yaml
│   ├── train_config_fast.yaml
│   └── inference_config.yaml
├── data/                  # 数据目录
├── checkpoints/           # 模型检查点
├── results/               # 结果输出
├── logs/                  # 日志文件
├── examples/              # 示例数据
├── train.py               # 训练脚本
├── train_classifier.py    # 下游分类实验
├── evaluate.py            # 质量评估脚本
├── requirements.txt       # 依赖项
├── README.md              # 本文件
├── SETUP.md               # 详细安装指南
└── SERVER_SETUP.md        # 服务器配置指南
```

**注意**：旧的 `codes/` 目录（来自 RetinaLogos 项目）已被清理，所有功能已整合到 `src/` 目录中。

## 🚀 安装

### 环境要求

- Python 3.10+
- CUDA 11.8+ (推荐 A100 GPU)
- PyTorch 2.0+
- 服务器路径限制：仅能在 `/home/Backup/maziheng` 操作（如适用）

### 快速安装

**详细的服务器环境配置请参考 [SERVER_SETUP.md](SERVER_SETUP.md)**

1. **检查依赖**：
```bash
python scripts/check_dependencies.py
```

2. **安装依赖**：
```bash
pip install -r requirements.txt

# 手动安装 flash-attn（可选，但强烈推荐）
pip install flash-attn --no-build-isolation
```

3. **验证安装**：
```bash
python scripts/smoke_test.py
```

**注意**：
- 手动安装依赖可以避免版本冲突
- Flash Attention 2 需要编译，可能需要 5-10 分钟
- 如果 flash-attn 安装失败，可以跳过（性能会降低 20-30%）

### 详细安装指南

- **服务器环境配置**：参考 [SERVER_SETUP.md](SERVER_SETUP.md)
- **依赖管理和配置**：参考 [SETUP.md](SETUP.md)

## ⚡ 快速启动

### 最简化的启动步骤

1. **克隆项目**（如果在服务器上）：
```bash
cd /home/Backup/maziheng
git clone <repo_url> medical-image-augmentation-system
cd medical-image-augmentation-system
```

2. **检查和安装依赖**：
```bash
# 检查已安装的包
python scripts/check_dependencies.py

# 安装缺失的包
pip install -r requirements.txt
```

3. **验证配置**：
```bash
# 检查路径配置
python scripts/check_paths.py

# 验证训练配置
python scripts/validate_config.py
```

4. **运行冒烟测试**：
```bash
python scripts/smoke_test.py
```

5. **开始训练**：
```bash
# 使用默认配置
python train.py --config configs/train_config.yaml

# 或使用快速训练配置（用于测试）
python train.py --config configs/train_config_fast.yaml
```

### 快速生成图像

```python
from src.inference import ImageGenerator

# 加载模型
generator = ImageGenerator(
    checkpoint_path="checkpoints/best_model.pth",
    device="cuda"
)

# 生成图像
image = generator.generate(
    caption="Severe diabetic retinopathy",
    num_inference_steps=50,
    guidance_scale=7.5
)

# 保存图像
image.save("generated.png")
```

### 快速启动 Web 界面

```bash
python src/app/demo.py --checkpoint checkpoints/best_model.pth
```

然后访问 `http://localhost:7860`

## 📊 数据准备

### 推荐数据集：FGADR

1. **下载数据集**
   - GitHub: https://github.com/csyizhou/FGADR-2842-Dataset
   - 包含 1,842 张高清眼底图和像素级病灶分割掩码

2. **数据格式**

创建 JSONL 格式的数据清单：

```jsonl
{"image_path": "data/FGADR/images/001.png", "caption": "2", "mask_path": "data/FGADR/masks/001.png"}
{"image_path": "data/FGADR/images/002.png", "caption": "Mild diabetic retinopathy", "mask_path": "data/FGADR/masks/002.png"}
{"image_path": "data/FGADR/images/003.png", "caption": "3", "mask_path": null}
```

字段说明：
- `image_path`: 眼底图像路径
- `caption`: DR 分级（0-4）或文本描述
- `mask_path`: 分割掩码路径（可选，null 表示仅文本模式）

3. **标签转文本映射**

系统自动将 DR 分级转换为病理文本：
- 0 → "No diabetic retinopathy"
- 1 → "Mild non-proliferative diabetic retinopathy"
- 2 → "Moderate non-proliferative diabetic retinopathy"
- 3 → "Severe non-proliferative diabetic retinopathy"
- 4 → "Proliferative diabetic retinopathy"

## 🎓 训练

### 1. 配置训练参数

编辑 `configs/train_config.yaml`：

```yaml
data:
  train_data_path: "data/train"
  val_data_path: "data/val"
  image_size: 1024
  batch_size: 32  # A100 可以使用更大的批次
  num_workers: 8

training:
  num_epochs: 100
  learning_rate: 1.0e-4
  use_amp: true
  amp_dtype: "bfloat16"  # A100 最佳
```

### 2. 开始训练

```bash
python train.py --config configs/train_config.yaml
```

### 3. 从检查点恢复

```bash
python train.py --config configs/train_config.yaml --resume checkpoints/latest.pth
```

### 4. 训练监控

- 检查点保存在 `checkpoints/`
- 训练日志保存在 `logs/training_log.json`
- 最佳模型：`checkpoints/best_model.pth`

### 5. A100 优化建议

- **Flash Attention 2**：2-3x 训练加速
- **BF16 混合精度**：减少显存，提升速度
- **批次大小**：从 32 开始，可尝试更大
- **数据加载**：`num_workers=4-8` 充分利用 CPU

## 🎨 推理

### 使用 Python API

```python
from src.inference import ImageGenerator

# 加载模型
generator = ImageGenerator(
    checkpoint_path="checkpoints/best_model.pth",
    device="cuda"
)

# 生成图像
image = generator.generate(
    caption="Severe diabetic retinopathy",
    mask=None,  # 可选：提供分割掩码
    image_size=1024,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)

# 保存图像
image.save("generated.png")
```

## 🌐 Gradio 演示

### 启动 Web 界面

```bash
python src/app/demo.py --checkpoint checkpoints/best_model.pth
```

### 访问界面

打开浏览器访问：`http://localhost:7860`

### 功能

- 上传分割掩码（可选）
- 选择 DR 分级或输入自定义文本
- 调整生成参数（采样步数、引导强度、随机种子）
- 实时生成并下载结果

### 创建公开链接

```bash
python src/app/demo.py --checkpoint checkpoints/best_model.pth --share
```

## 📈 下游评估

### 质量评估（PSNR, SSIM）

评估生成图像与参考图像的质量：

```bash
python evaluate.py \
    --generated results/generated_images/ \
    --reference data/reference_images/ \
    --output results/evaluation_results.json
```

结果包含：
- **PSNR** (Peak Signal-to-Noise Ratio)：越高越好，通常 >30dB 表示高质量
- **SSIM** (Structural Similarity Index)：范围 0-1，越接近 1 越好
- **MAE** (Mean Absolute Error)：越低越好
- **MSE** (Mean Squared Error)：越低越好

示例输出：
```
EVALUATION SUMMARY
======================================================================
Number of images: 100

PSNR: 32.45 ± 2.31 dB
  Range: [28.12, 36.78]

SSIM: 0.8923 ± 0.0456
  Range: [0.7834, 0.9512]

MAE: 12.34 ± 3.21
MSE: 234.56 ± 45.67
======================================================================
```

### 分类实验（证明增广价值）

使用 ResNet-50 进行 DR 分级分类：

```bash
# 实验 1：仅原始数据
python train_classifier.py \
    --original_data data/train_manifest.jsonl \
    --val_data data/val_manifest.jsonl \
    --epochs 20

# 实验 2：原始 + 增广数据
python train_classifier.py \
    --original_data data/train_manifest.jsonl \
    --augmented_data data/augmented_manifest.jsonl \
    --val_data data/val_manifest.jsonl \
    --epochs 20
```

### 结果分析

结果保存在 `results/downstream_evaluation.json`：

```json
{
  "original_only": {
    "best_val_acc": 0.7234
  },
  "original_plus_augmented": {
    "best_val_acc": 0.7456
  },
  "improvement_percent": 2.22
}
```

准确率提升 2%+ 证明增广系统的价值！

## 💡 简化说明

### 设计原则：避免过度工程化

本项目采用**实用主义**的开发方法，专注于核心功能的实现，避免不必要的复杂性：

#### 1. 手动依赖管理

**为什么不用自动化安装脚本？**
- ✅ 手动安装可以避免版本冲突
- ✅ 可以利用服务器上已安装的包
- ✅ 更容易排查和解决依赖问题
- ❌ 自动化脚本可能引入不必要的包或版本冲突

**实践**：
```bash
# 先检查已安装的包
python scripts/check_dependencies.py

# 根据报告手动安装缺失的包
pip install <missing_package>
```

#### 2. 简化的测试策略

**为什么不用复杂的测试框架？**
- ✅ 冒烟测试快速验证核心功能（< 2 分钟）
- ✅ 简单的单元测试覆盖关键函数
- ✅ 专注实用性，不追求 100% 覆盖率
- ❌ 属性测试（Hypothesis）对单人毕设项目过于复杂
- ❌ 集成测试在快速迭代阶段不必要

**实践**：
```bash
# 快速验证核心功能
python scripts/smoke_test.py

# 运行简单的单元测试
python -m pytest tests/
```

#### 3. 相对路径优先

**为什么使用相对路径？**
- ✅ 项目可移植性更好
- ✅ 避免硬编码绝对路径
- ✅ 符合服务器路径限制
- ❌ 绝对路径在不同环境下容易出错

**实践**：
```yaml
# 配置文件中使用相对路径
data:
  train_data: "data/train.jsonl"  # ✅ 相对路径
  # train_data: "/home/user/data/train.jsonl"  # ❌ 绝对路径
```

#### 4. 配置验证而非自动修复

**为什么不自动修复配置？**
- ✅ 让开发者了解配置的含义
- ✅ 避免自动修复引入新问题
- ✅ 提供明确的修改建议
- ❌ 自动修复可能掩盖潜在问题

**实践**：
```bash
# 验证配置并获取建议
python scripts/validate_config.py

# 根据建议手动调整配置
vim configs/train_config.yaml
```

#### 5. 安全的清理流程

**为什么不直接删除旧代码？**
- ✅ 先重命名，测试通过后再删除
- ✅ 提供回滚机制
- ✅ 记录所有操作日志
- ❌ 直接删除可能导致项目崩溃

**实践**：
```bash
# 安全重命名
mv codes codes_backup

# 运行测试
python scripts/smoke_test.py

# 测试通过后删除
rm -rf codes_backup

# 如果测试失败，回滚
python scripts/rollback.py
```

### 核心理念

1. **简单优于复杂**：能用简单方法解决的，不用复杂方案
2. **手动优于自动**：关键操作手动执行，确保可控
3. **实用优于完美**：专注核心功能，不追求过度优化
4. **安全优于快速**：宁可多一步验证，不冒险直接操作

### 适用场景

本项目的简化方法适用于：
- ✅ 单人开发的毕设项目
- ✅ 快速迭代的研究原型
- ✅ 明确需求的特定任务
- ❌ 大型团队协作项目
- ❌ 需要高可靠性的生产环境
- ❌ 复杂的多模块系统

## 🔬 技术细节

### 模型架构

- **基础模型**：NextDiT (Diffusion Transformer)
- **扩展**：通道拼接支持掩码输入 (RGB + Mask)
- **参数量**：~2B (2304 hidden dim, 24 layers)

### 训练方法

- **Loss**：Flow Matching / Rectified Flow (Velocity Prediction)
- **优化器**：AdamW
- **学习率调度**：Cosine Annealing
- **混合精度**：BF16 (A100) / FP16

### 采样方法

- **ODE Solver**：Euler method
- **步数**：50-100 steps
- **CFG**：Classifier-Free Guidance (scale 7-10)

## 📝 引用

如果使用本项目，请引用：

```bibtex
@misc{medical-image-augmentation,
  title={Medical Image Augmentation System for Diabetic Retinopathy},
  author={Your Name},
  year={2026}
}
```

基于 RetinaLogos 项目：
- Github: https://github.com/uni-medical/retina-text2cfp
- GitHub: https://github.com/Alpha-VLLM/Lumina-T2X


## ⚠️ 免责声明

本系统为研究原型，生成的图像**不应用于临床诊断**。仅供学术研究和教育用途。

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系作者。

## 📄 许可证

本项目遵循 MIT 许可证。详见 LICENSE 文件。
