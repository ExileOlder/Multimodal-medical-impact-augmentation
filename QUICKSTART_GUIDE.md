# 🚀 快速开始指南

本指南将指导你完成从环境配置到训练、推理、评估的完整流程。

## 📋 目录

1. [环境配置](#1-环境配置)
2. [数据准备](#2-数据准备)
3. [验证安装](#3-验证安装)
4. [模型训练](#4-模型训练)
5. [图像生成](#5-图像生成)
6. [质量评估](#6-质量评估)
7. [下游分类实验](#7-下游分类实验)
8. [常见问题](#8-常见问题)

---

## 1. 环境配置

### 步骤 1.1：检查系统要求

确保你的服务器满足以下要求：
- Python 3.10+
- CUDA 11.8+ (A100 GPU)
- 至少 40GB GPU 显存
- 至少 100GB 磁盘空间

检查 CUDA 版本：
```bash
nvidia-smi
nvcc --version
```

### 步骤 1.2：创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n medical-aug python=3.10
conda activate medical-aug

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 步骤 1.3：安装依赖

```bash
# 安装 PyTorch (根据你的 CUDA 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

**注意**：Flash Attention 2 的安装可能需要 5-10 分钟编译时间。如果编译失败：
```bash
# 可以先注释掉 requirements.txt 中的 flash-attn
# 系统仍可运行，只是训练速度会慢一些
```

### 步骤 1.4：验证 PyTorch 和 CUDA

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

预期输出：
```
PyTorch: 2.x.x
CUDA available: True
CUDA version: 11.8
```

---

## 2. 数据准备

### 步骤 2.1：下载 FGADR 数据集

1. 访问 GitHub: https://github.com/csyizhou/FGADR-2842-Dataset
2. 下载数据集（约 2GB）
3. 解压到 `data/FGADR/` 目录

目录结构应该是：
```
data/FGADR/
├── Seg-set/
│   ├── Original_Images/
│   │   ├── 1_left.png
│   │   ├── 1_right.png
│   │   └── ...
│   └── Lesion_Masks/
│       ├── HardExudates/
│       ├── Haemorrhages/
│       ├── Microaneurysms/
│       └── SoftExudates/
└── DR_Grading/
    └── DR_grading.csv
```

### 步骤 2.2：创建 JSONL 数据清单

创建一个 Python 脚本来生成 JSONL 文件：

```bash
# 创建 prepare_data.py
cat > prepare_data.py << 'EOF'
"""准备 FGADR 数据集的 JSONL 清单文件"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image

def merge_lesion_masks(image_id, mask_dir):
    """合并多个病灶掩码为单个掩码"""
    lesion_types = ['HardExudates', 'Haemorrhages', 'Microaneurysms', 'SoftExudates']
    
    merged_mask = None
    
    for lesion_type in lesion_types:
        mask_path = mask_dir / lesion_type / f"{image_id}.png"
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'))
            if merged_mask is None:
                merged_mask = mask
            else:
                merged_mask = np.maximum(merged_mask, mask)
    
    return merged_mask

def create_jsonl_manifest():
    """创建训练和验证数据的 JSONL 清单"""
    
    # 路径配置
    base_dir = Path("data/FGADR")
    image_dir = base_dir / "Seg-set" / "Original_Images"
    mask_dir = base_dir / "Seg-set" / "Lesion_Masks"
    grading_file = base_dir / "DR_Grading" / "DR_grading.csv"
    
    # 创建输出目录
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # 读取 DR 分级
    df = pd.read_csv(grading_file)
    
    # 创建合并掩码目录
    merged_mask_dir = base_dir / "Merged_Masks"
    merged_mask_dir.mkdir(exist_ok=True)
    
    train_data = []
    val_data = []
    
    print("Processing images...")
    
    for idx, row in df.iterrows():
        image_name = row['image_id']  # 例如: "1_left"
        dr_grade = int(row['DR_grade'])
        
        image_path = image_dir / f"{image_name}.png"
        
        if not image_path.exists():
            continue
        
        # 合并病灶掩码
        merged_mask = merge_lesion_masks(image_name, mask_dir)
        
        if merged_mask is not None:
            # 保存合并后的掩码
            mask_path = merged_mask_dir / f"{image_name}_mask.png"
            Image.fromarray(merged_mask).save(mask_path)
            mask_path_str = str(mask_path)
        else:
            mask_path_str = None
        
        entry = {
            "image_path": str(image_path),
            "caption": str(dr_grade),  # 将自动转换为文本
            "mask_path": mask_path_str,
            "label": dr_grade  # 用于下游分类
        }
        
        # 80/20 划分训练集和验证集
        if idx % 5 == 0:
            val_data.append(entry)
        else:
            train_data.append(entry)
    
    # 保存 JSONL 文件
    train_file = output_dir / "train_manifest.jsonl"
    val_file = output_dir / "val_manifest.jsonl"
    
    with open(train_file, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
    
    with open(val_file, 'w') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n✓ Created {train_file} with {len(train_data)} entries")
    print(f"✓ Created {val_file} with {len(val_data)} entries")
    print(f"✓ Merged masks saved to {merged_mask_dir}")

if __name__ == "__main__":
    create_jsonl_manifest()
EOF

# 运行脚本
python prepare_data.py
```

### 步骤 2.3：验证数据

```bash
# 检查生成的文件
ls -lh data/*.jsonl
head -n 3 data/train_manifest.jsonl
```

---

## 3. 验证安装

### 步骤 3.1：测试数据模块

```bash
python test_data_module.py
```

预期输出：
```
======================================================================
DATA PROCESSING MODULE VALIDATION
======================================================================

==================================================
Testing Label-to-Caption Conversion
==================================================
Grade 0: No diabetic retinopathy
Grade 1: Mild non-proliferative diabetic retinopathy
...
✓ Label-to-caption conversion working

✓ ALL TESTS PASSED - Data processing module is working correctly!
```

### 步骤 3.2：测试模型扩展

```bash
python test_model_extension.py
```

预期输出：
```
======================================================================
MODEL EXTENSION VALIDATION
======================================================================

==================================================
Testing Model Initialization
==================================================
Model created successfully
Parameter count: 2,304,000,000
✓ Model initialization working

✓ ALL TESTS PASSED - Model extension is working correctly!
```

---

## 4. 模型训练

### 步骤 4.1：调整训练配置

编辑 `configs/train_config.yaml`：

```yaml
data:
  train_data_path: "data"  # 包含 train_manifest.jsonl 的目录
  val_data_path: "data"    # 包含 val_manifest.jsonl 的目录
  image_size: 512          # 先用 512 测试，稳定后改为 1024
  batch_size: 8            # 根据显存调整（A100 可用 16-32）
  num_workers: 4

training:
  num_epochs: 50           # 先训练 50 个 epoch
  learning_rate: 1.0e-4
  use_amp: true
  amp_dtype: "bfloat16"    # A100 最佳
  save_every: 5
  log_every: 50
```

### 步骤 4.2：开始训练

```bash
# 启动训练
python train.py --config configs/train_config.yaml

# 如果需要后台运行
nohup python train.py --config configs/train_config.yaml > training.log 2>&1 &

# 查看日志
tail -f training.log
```

### 步骤 4.3：监控训练

训练过程中会显示：
```
======================================================================
STARTING TRAINING
======================================================================
Total epochs: 50
Device: cuda
Mixed precision: True (bfloat16)
Batch size: 8
======================================================================

Epoch 1/50: 100%|████████| 150/150 [05:23<00:00, loss=0.1234]

Epoch 1/50
  Train Loss: 0.1234
  Learning Rate: 0.000100

Saved checkpoint: checkpoints/checkpoint_epoch_5.pth
```

### 步骤 4.4：从检查点恢复（如果中断）

```bash
python train.py --config configs/train_config.yaml --resume checkpoints/latest.pth
```

### 步骤 4.5：训练时间估算

- **A100 GPU**：
  - 512x512 图像：约 2-3 小时/epoch（1000 张图）
  - 1024x1024 图像：约 5-6 小时/epoch
- **总训练时间**：50 epochs × 3 小时 = 约 150 小时（6-7 天）

**建议**：
1. 先用 512 分辨率训练 10-20 epochs 验证流程
2. 确认无误后再用 1024 分辨率完整训练

---

## 5. 图像生成

### 步骤 5.1：使用 Gradio 界面（推荐）

```bash
# 启动 Gradio 应用
python src/app/demo.py --checkpoint checkpoints/best_model.pth

# 如果在服务器上，创建公开链接
python src/app/demo.py --checkpoint checkpoints/best_model.pth --share
```

访问显示的 URL（例如 `http://localhost:7860`）

**界面操作**：
1. 上传分割掩码（可选）
2. 选择 DR 分级（0-4）或输入自定义文本
3. 调整参数：
   - 采样步数：50（推荐）
   - 引导强度：7.5（推荐）
   - 随机种子：42（可复现）
4. 点击"生成图像"
5. 下载生成结果

### 步骤 5.2：批量生成（Python 脚本）

创建批量生成脚本：

```bash
cat > batch_generate.py << 'EOF'
"""批量生成图像"""

from src.inference import ImageGenerator, save_batch_results
from PIL import Image

# 加载生成器
generator = ImageGenerator(
    checkpoint_path="checkpoints/best_model.pth",
    device="cuda"
)

# 准备输入
captions = [
    "No diabetic retinopathy",
    "Mild diabetic retinopathy",
    "Moderate diabetic retinopathy",
    "Severe diabetic retinopathy",
    "Proliferative diabetic retinopathy"
]

# 批量生成
print("Generating images...")
images = generator.batch_generate(
    captions=captions,
    masks=None,
    image_size=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)

# 保存结果
metadata_list = [{"caption": cap, "seed": 42} for cap in captions]
results = save_batch_results(
    images=images,
    output_dir="results",
    metadata_list=metadata_list
)

print(f"\n✓ Generated {len(images)} images")
print(f"✓ Saved to: results/")
EOF

python batch_generate.py
```

---

## 6. 质量评估

### 步骤 6.1：准备参考图像

确保你有：
- `results/generated/` - 生成的图像
- `data/reference/` - 对应的参考图像（原始图像）

### 步骤 6.2：运行评估

```bash
python evaluate.py \
    --generated results/generated/ \
    --reference data/reference/ \
    --output results/evaluation_results.json
```

### 步骤 6.3：查看结果

```bash
# 查看 JSON 结果
cat results/evaluation_results.json | python -m json.tool

# 或直接查看摘要
python -c "
import json
with open('results/evaluation_results.json') as f:
    data = json.load(f)
    summary = data['summary']
    print(f\"PSNR: {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f} dB\")
    print(f\"SSIM: {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}\")
"
```

### 步骤 6.4：解读指标

**PSNR (Peak Signal-to-Noise Ratio)**：
- > 30 dB：高质量
- 25-30 dB：中等质量
- < 25 dB：低质量

**SSIM (Structural Similarity Index)**：
- > 0.9：非常相似
- 0.8-0.9：相似
- < 0.8：差异较大

---

## 7. 下游分类实验

### 步骤 7.1：准备分类数据

确保 JSONL 文件包含 `label` 字段（DR 分级 0-4）

### 步骤 7.2：实验 1 - 仅原始数据

```bash
python train_classifier.py \
    --original_data data/train_manifest.jsonl \
    --val_data data/val_manifest.jsonl \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4
```

### 步骤 7.3：生成增广数据

使用 Gradio 或批量脚本生成增广图像，并创建增广数据清单：

```bash
cat > create_augmented_manifest.py << 'EOF'
"""创建增广数据清单"""

import json
from pathlib import Path

augmented_dir = Path("results/generated")
output_file = Path("data/augmented_manifest.jsonl")

entries = []
for img_path in augmented_dir.glob("*.png"):
    # 从 metadata 读取标签
    metadata_path = img_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            label = metadata.get('dr_grade', 0)
    else:
        label = 0  # 默认
    
    entry = {
        "image_path": str(img_path),
        "caption": str(label),
        "mask_path": None,
        "label": int(label)
    }
    entries.append(entry)

with open(output_file, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

print(f"✓ Created {output_file} with {len(entries)} entries")
EOF

python create_augmented_manifest.py
```

### 步骤 7.4：实验 2 - 原始 + 增广数据

```bash
python train_classifier.py \
    --original_data data/train_manifest.jsonl \
    --augmented_data data/augmented_manifest.jsonl \
    --val_data data/val_manifest.jsonl \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4
```

### 步骤 7.5：查看对比结果

```bash
cat results/downstream_evaluation.json | python -m json.tool
```

预期输出：
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

**准确率提升 2%+ 证明增广系统的价值！**

---

## 8. 常见问题

### Q1: CUDA Out of Memory

**解决方案**：
```yaml
# 减小批次大小
batch_size: 4  # 从 8 或 16 减小

# 或减小图像分辨率
image_size: 512  # 从 1024 减小
```

### Q2: Flash Attention 编译失败

**解决方案**：
```bash
# 注释掉 requirements.txt 中的 flash-attn
# 系统仍可运行，只是训练速度会慢一些
```

### Q3: 训练 Loss 不下降

**检查**：
1. 学习率是否合适（尝试 1e-5 到 1e-3）
2. 数据是否正确加载（检查日志）
3. 是否使用了正确的 Loss（Flow Matching）

### Q4: 生成图像质量差

**可能原因**：
1. 训练不充分（需要更多 epochs）
2. 采样步数太少（增加到 100）
3. 引导强度不合适（尝试 5-10）

### Q5: Gradio 界面无法访问

**解决方案**：
```bash
# 检查端口是否被占用
netstat -tuln | grep 7860

# 更换端口
python src/app/demo.py --checkpoint checkpoints/best_model.pth --port 8080

# 如果在服务器上，使用 SSH 端口转发
ssh -L 7860:localhost:7860 user@server
```

---

## 📊 完整流程时间线

| 步骤 | 预计时间 | 说明 |
|------|---------|------|
| 环境配置 | 30 分钟 | 包括依赖安装 |
| 数据准备 | 1 小时 | 下载和处理 FGADR |
| 验证安装 | 10 分钟 | 运行测试脚本 |
| 模型训练 | 6-7 天 | 50 epochs @ 1024x1024 |
| 图像生成 | 1 小时 | 生成 100-500 张 |
| 质量评估 | 10 分钟 | PSNR/SSIM 计算 |
| 下游实验 | 4-6 小时 | 两组分类实验 |

**总计**：约 7-8 天（主要是训练时间）

---

## 🎯 答辩准备清单

- [ ] 训练完成，保存最佳模型
- [ ] 生成至少 100 张高质量图像
- [ ] 运行质量评估，获得 PSNR/SSIM 指标
- [ ] 完成下游分类实验，证明准确率提升
- [ ] 准备 Gradio 演示（实时生成）
- [ ] 准备 PPT：
  - 系统架构图
  - 生成结果对比
  - 质量指标图表
  - 下游评估结果
  - 技术亮点（Flow Matching, A100 优化）

---

## 📞 需要帮助？

如果遇到问题：
1. 检查日志文件：`logs/training_log.json`
2. 查看错误信息
3. 参考 README.md 和 PROJECT_SUMMARY.md
4. 检查 GitHub Issues（如果有）

祝你毕设顺利！🎉
