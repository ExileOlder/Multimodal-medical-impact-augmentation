# 🏥 Medical Image Augmentation System

基于 RetinaLogos 的医学影像增广系统，支持文本和分割掩码条件的糖尿病视网膜病变（DR）眼底图像生成。

## ✨ 特性

- **多模态条件生成**：支持文本描述和分割掩码条件
- **Flow Matching 训练**：使用 Rectified Flow 进行高效训练
- **A100 优化**：Flash Attention 2 + BF16 混合精度训练
- **Gradio 演示界面**：简洁易用的 Web 界面
- **下游任务评估**：ResNet-50 分类实验验证增广价值

## 📋 目录

- [安装](#安装)
- [数据准备](#数据准备)
- [训练](#训练)
- [推理](#推理)
- [Gradio 演示](#gradio-演示)
- [下游评估](#下游评估)
- [项目结构](#项目结构)

## 🚀 安装

### 1. 环境要求

- Python 3.10+
- CUDA 11.8+ (推荐 A100 GPU)
- PyTorch 2.0+

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：Flash Attention 2 需要 CUDA 并可能需要编译时间。如果遇到问题：
- 确保已安装 CUDA toolkit
- A100 服务器应该能成功编译
- 如果编译失败，可以注释掉 `requirements.txt` 中的 `flash-attn` 继续使用

### 3. 验证安装

```bash
python test_data_module.py
python test_model_extension.py
```

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

## 📁 项目结构

```
.
├── src/
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
├── codes/                 # 原始 RetinaLogos 代码
│   ├── models/
│   └── transport/         # Flow Matching 实现
├── configs/               # 配置文件
│   ├── train_config.yaml
│   └── inference_config.yaml
├── train.py               # 训练脚本
├── train_classifier.py    # 下游分类实验
├── evaluate.py            # 质量评估脚本
├── test_data_module.py    # 数据模块测试
├── test_model_extension.py # 模型测试
└── requirements.txt       # 依赖项
```

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
