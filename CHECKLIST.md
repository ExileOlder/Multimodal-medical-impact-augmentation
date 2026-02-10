# ✅ 毕设完成检查清单

使用本清单确保所有步骤都已完成。

## 📋 第一阶段：环境准备（第 1 天）

- [ ] **1.1** 检查服务器配置（A100 GPU, CUDA 11.8+）
- [ ] **1.2** 创建 Python 虚拟环境
- [ ] **1.3** 安装 PyTorch 和 CUDA
- [ ] **1.4** 安装项目依赖（`pip install -r requirements.txt`）
- [ ] **1.5** 验证 PyTorch 和 CUDA 可用
- [ ] **1.6** 运行 `python test_data_module.py` ✓
- [ ] **1.7** 运行 `python test_model_extension.py` ✓

**预期结果**：所有测试通过，显示 "ALL TESTS PASSED"

---

## 📊 第二阶段：数据准备（第 1-2 天）

- [ ] **2.1** 下载 FGADR 数据集（~2GB）
- [ ] **2.2** 解压到 `data/FGADR/` 目录
- [ ] **2.3** 创建 `prepare_data.py` 脚本
- [ ] **2.4** 运行数据准备脚本
- [ ] **2.5** 验证生成的 JSONL 文件：
  - [ ] `data/train_manifest.jsonl` 存在
  - [ ] `data/val_manifest.jsonl` 存在
  - [ ] 文件包含正确的字段（image_path, caption, mask_path, label）
- [ ] **2.6** 检查合并后的掩码：`data/FGADR/Merged_Masks/`

**预期结果**：
- 训练集：~1400 条数据
- 验证集：~350 条数据

---

## 🎓 第三阶段：模型训练（第 2-9 天）

### 初步测试（第 2 天）

- [ ] **3.1** 编辑 `configs/train_config.yaml`：
  - [ ] `image_size: 512`（测试用）
  - [ ] `batch_size: 8`
  - [ ] `num_epochs: 5`（快速测试）
- [ ] **3.2** 运行测试训练：`python train.py --config configs/train_config.yaml`
- [ ] **3.3** 验证训练可以正常运行（至少 1 个 epoch）
- [ ] **3.4** 检查生成的文件：
  - [ ] `checkpoints/checkpoint_epoch_5.pth`
  - [ ] `logs/training_log.json`

### 完整训练（第 3-9 天）

- [ ] **3.5** 更新配置为完整训练：
  - [ ] `image_size: 1024`
  - [ ] `batch_size: 16-32`（根据显存）
  - [ ] `num_epochs: 50`
- [ ] **3.6** 启动完整训练（后台运行）：
  ```bash
  nohup python train.py --config configs/train_config.yaml > training.log 2>&1 &
  ```
- [ ] **3.7** 定期检查训练进度：
  - [ ] 每天查看 `tail -f training.log`
  - [ ] 确认 Loss 在下降
  - [ ] 检查检查点文件定期保存
- [ ] **3.8** 训练完成后，确认文件存在：
  - [ ] `checkpoints/best_model.pth`
  - [ ] `checkpoints/latest.pth`
  - [ ] `logs/training_log.json`

**预期结果**：
- 训练 Loss 从 ~0.5 降到 ~0.05
- 总训练时间：约 6-7 天

---

## 🎨 第四阶段：图像生成（第 10 天）

### Gradio 界面测试

- [ ] **4.1** 启动 Gradio 应用：
  ```bash
  python src/app/demo.py --checkpoint checkpoints/best_model.pth
  ```
- [ ] **4.2** 访问界面（`http://localhost:7860`）
- [ ] **4.3** 测试生成功能：
  - [ ] 选择 DR 分级 0（无 DR）
  - [ ] 点击生成，等待结果
  - [ ] 验证生成的图像质量
- [ ] **4.4** 测试不同参数：
  - [ ] DR 分级 1-4
  - [ ] 不同采样步数（25, 50, 100）
  - [ ] 不同引导强度（5, 7.5, 10）
- [ ] **4.5** 保存生成的图像用于展示

### 批量生成

- [ ] **4.6** 创建 `batch_generate.py` 脚本
- [ ] **4.7** 批量生成至少 100 张图像
- [ ] **4.8** 检查生成结果：`results/batch_*/`
- [ ] **4.9** 选择 10-20 张最佳图像用于答辩

**预期结果**：
- 生成图像视觉质量良好
- 符合指定的 DR 分级特征

---

## 📈 第五阶段：质量评估（第 10 天）

- [ ] **5.1** 准备参考图像目录：`data/reference/`
- [ ] **5.2** 运行质量评估：
  ```bash
  python evaluate.py \
      --generated results/generated/ \
      --reference data/reference/ \
      --output results/evaluation_results.json
  ```
- [ ] **5.3** 查看评估结果：
  - [ ] PSNR 值（目标 > 30 dB）
  - [ ] SSIM 值（目标 > 0.85）
- [ ] **5.4** 保存结果用于答辩

**预期结果**：
- PSNR: 30-35 dB
- SSIM: 0.85-0.95

---

## 🔬 第六阶段：下游分类实验（第 11 天）

### 实验 1：仅原始数据

- [ ] **6.1** 运行基线实验：
  ```bash
  python train_classifier.py \
      --original_data data/train_manifest.jsonl \
      --val_data data/val_manifest.jsonl \
      --epochs 20
  ```
- [ ] **6.2** 记录准确率（例如：72.34%）

### 实验 2：原始 + 增广数据

- [ ] **6.3** 创建增广数据清单：`data/augmented_manifest.jsonl`
- [ ] **6.4** 运行增广实验：
  ```bash
  python train_classifier.py \
      --original_data data/train_manifest.jsonl \
      --augmented_data data/augmented_manifest.jsonl \
      --val_data data/val_manifest.jsonl \
      --epochs 20
  ```
- [ ] **6.5** 记录准确率（例如：74.56%）
- [ ] **6.6** 计算提升（例如：+2.22%）
- [ ] **6.7** 查看详细结果：`results/downstream_evaluation.json`

**预期结果**：
- 准确率提升 2%+
- 证明增广系统的价值

---

## 📊 第七阶段：答辩准备（第 12-13 天）

### 材料准备

- [ ] **7.1** 整理生成结果：
  - [ ] 选择 10-20 张最佳生成图像
  - [ ] 准备原始图像对比
  - [ ] 制作对比图（原始 vs 生成）

- [ ] **7.2** 准备质量指标图表：
  - [ ] PSNR 柱状图
  - [ ] SSIM 柱状图
  - [ ] 训练 Loss 曲线

- [ ] **7.3** 准备下游评估结果：
  - [ ] 准确率对比柱状图
  - [ ] 混淆矩阵
  - [ ] 分类报告

- [ ] **7.4** 准备系统架构图：
  - [ ] 数据流程图
  - [ ] 模型结构图
  - [ ] 训练流程图

### PPT 制作

- [ ] **7.5** 封面：题目、姓名、导师
- [ ] **7.6** 研究背景与意义
- [ ] **7.7** 相关工作（RetinaLogos, Lumina-Next）
- [ ] **7.8** 系统设计：
  - [ ] 整体架构
  - [ ] 数据处理流程
  - [ ] 模型扩展（掩码条件）
  - [ ] 训练方法（Flow Matching）
- [ ] **7.9** 实验结果：
  - [ ] 生成结果展示
  - [ ] 质量评估（PSNR, SSIM）
  - [ ] 下游分类实验
- [ ] **7.10** 技术亮点：
  - [ ] Flow Matching Loss
  - [ ] A100 优化（Flash Attention 2, BF16）
  - [ ] 标签转文本功能
- [ ] **7.11** 总结与展望
- [ ] **7.12** 致谢

### 演示准备

- [ ] **7.13** 准备 Gradio 演示：
  - [ ] 测试界面流畅运行
  - [ ] 准备演示用例（不同 DR 分级）
  - [ ] 准备备用截图（防止网络问题）

- [ ] **7.14** 准备答辩讲稿：
  - [ ] 5-10 分钟口述内容
  - [ ] 预演 2-3 次
  - [ ] 准备常见问题回答

---

## 🎯 最终检查（答辩前 1 天）

- [ ] **8.1** 所有代码文件完整
- [ ] **8.2** 所有结果文件保存
- [ ] **8.3** PPT 制作完成
- [ ] **8.4** Gradio 演示可正常运行
- [ ] **8.5** 准备 U 盘备份（代码 + 结果 + PPT）
- [ ] **8.6** 打印必要材料（如需要）
- [ ] **8.7** 预演答辩流程

---

## 📁 交付物清单

### 代码

- [ ] 完整的项目代码（GitHub 仓库或压缩包）
- [ ] README.md（使用说明）
- [ ] requirements.txt（依赖列表）
- [ ] 配置文件（configs/*.yaml）

### 模型

- [ ] 训练好的模型（checkpoints/best_model.pth）
- [ ] 训练日志（logs/training_log.json）

### 结果

- [ ] 生成的图像（至少 100 张）
- [ ] 质量评估结果（evaluation_results.json）
- [ ] 下游分类结果（downstream_evaluation.json）

### 文档

- [ ] 毕业论文（Word/PDF）
- [ ] 答辩 PPT
- [ ] 项目说明文档

---

## 🚨 常见问题快速解决

### 训练相关

- **Loss 不下降**：检查学习率，尝试 1e-5 到 1e-3
- **CUDA OOM**：减小 batch_size 或 image_size
- **训练太慢**：确认 Flash Attention 已安装，使用 BF16

### 生成相关

- **生成质量差**：增加采样步数到 100，调整引导强度
- **生成速度慢**：减小图像尺寸，使用更少的采样步数

### 评估相关

- **PSNR 太低**：可能是训练不充分，继续训练
- **准确率提升不明显**：生成更多增广数据，确保数据多样性

---

## 🎉 完成标志

当你完成以下所有项目时，你的毕设就完成了：

✅ 模型训练完成，Loss 收敛
✅ 生成至少 100 张高质量图像
✅ PSNR > 30 dB, SSIM > 0.85
✅ 下游分类准确率提升 2%+
✅ Gradio 演示可正常运行
✅ PPT 制作完成
✅ 答辩讲稿准备完毕

**恭喜你！准备好答辩吧！** 🎓🎉
