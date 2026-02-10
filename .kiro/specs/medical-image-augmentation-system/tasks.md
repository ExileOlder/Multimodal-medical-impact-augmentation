# Implementation Plan: Medical Image Augmentation System (Simplified)

## Overview

本实现计划将医学影像增广系统的设计转化为可执行的编码任务。系统基于RetinaLogos项目扩展，添加分割掩码条件支持，并使用Gradio提供简洁的演示界面。实现采用渐进式方法，每个任务都在前一个任务的基础上构建，确保代码始终处于可运行状态。

**核心简化**：使用Gradio替代Flask全栈开发，大幅减少Web开发工作量（从500-800行代码降至30-50行）。

## Tasks

- [x] 1. 项目结构初始化与环境配置
  - 创建项目目录结构（src/data, src/models, src/training, src/inference, src/evaluation, src/app）
  - 创建requirements.txt文件，包含所有依赖项（PyTorch, Gradio, Pillow, NumPy, pytest等，移除Flask相关）
  - **⚡ A100优化：添加flash-attn依赖以启用Flash Attention 2，可将训练速度提升2-3倍**
  - 创建配置文件模板（configs/train_config.yaml, configs/inference_config.yaml）
  - 设置日志目录和输出目录结构（checkpoints/, results/, logs/, examples/）
  - _Requirements: 4.1, 10.3_

- [-] 2. 数据预处理模块实现
  - [x] 2.1 实现JSONL数据加载器
    - 编写load_jsonl函数，解析JSONL文件并提取image_path、caption、mask_path字段
    - 处理缺失mask_path的情况，标记为仅文本模式
    - 支持批量加载文件夹中的所有JSONL文件
    - _Requirements: 1.1, 1.2, 1.4_
  
  - [ ]* 2.2 编写属性测试：JSONL解析完整性
    - **Property 1: JSONL解析完整性**
    - **Validates: Requirements 1.1**
  
  - [ ]* 2.3 编写属性测试：缺失掩码处理
    - **Property 2: 缺失掩码处理**
    - **Validates: Requirements 1.2, 3.4**
  
  - [ ]* 2.4 编写属性测试：批量数据加载
    - **Property 4: 批量数据加载**
    - **Validates: Requirements 1.4**
  
  - [x] 2.5 实现图像预处理函数
    - 编写preprocess_image函数：调整图像到目标分辨率，执行像素值归一化
    - 编写preprocess_mask函数：调整掩码尺寸，转换为整数类别标签格式
    - **实现标签转文本（Label-to-Caption）功能**：将DR分级标签（0-4）转换为病理文本描述（如"Severe non-proliferative diabetic retinopathy"），完成"病理文本整理"要求
    - 实现数据统计日志输出功能
    - _Requirements: 2.1, 2.2, 2.3, 1.3, 1.5_
  
  - [ ]* 2.6 编写属性测试：图像预处理
    - **Property 6: 图像分辨率标准化**
    - **Property 7: 像素值归一化范围**
    - **Property 8: 掩码格式转换**
    - **Validates: Requirements 2.1, 2.2, 2.3**
  
  - [ ]* 2.7 编写属性测试：掩码尺寸调整
    - **Property 3: 掩码尺寸自动调整**
    - **Validates: Requirements 1.3**
  
  - [x] 2.8 实现PyTorch Dataset类
    - 创建MultimodalDataset类，继承torch.utils.data.Dataset
    - 实现__getitem__方法，返回图像、文本、掩码三元组
    - 实现collate_fn函数用于批次数据整理
    - _Requirements: 1.1, 1.2, 1.3_

- [x] 3. 检查点 - 数据处理模块验证
  - 使用小规模测试数据验证数据加载和预处理功能
  - 确保能够正确读取图像和对应掩码
  - 如有问题请向用户询问

- [-] 4. 掩码处理与模型扩展
  - [x] 4.1 实现掩码处理工具函数
    - 编写prepare_mask函数：分辨率调整（使用F.interpolate）
    - 简单的通道维度处理，无需复杂的One-Hot编码
    - 支持零张量输入（无掩码情况）
    - _Requirements: 3.1_
  
  - [ ]* 4.2 编写属性测试：掩码处理输出维度
    - **Property 10: 掩码处理输出维度**
    - **Validates: Requirements 3.1**
  
  - [x] 4.3 扩展NexDiT模型支持掩码输入
    - 复制codes/models/model.py为src/models/nexdit_mask.py
    - 修改NextDiT类的__init__方法，添加mask_channels参数（默认为1）
    - 修改输入层通道数为in_channels + mask_channels
    - 修改forward方法，在patchify前进行通道拼接：torch.cat([x, condition_mask], dim=1)
    - 处理condition_mask为None的情况，使用零张量替代
    - **⚡ A100优化：保留原代码中的flash_attn_varlen_func相关逻辑，确保Flash Attention 2正常工作**
    - _Requirements: 3.2, 3.3, 3.4_
  
  - [ ]* 4.4 编写属性测试：模型多模态输入
    - **Property 11: 模型多模态输入接口**
    - **Property 12: 通道拼接正确性**
    - **Validates: Requirements 3.2, 3.3**
  
  - [ ]* 4.5 编写单元测试：模型前向传播
    - 测试模型能够接收文本、图像、掩码输入并返回正确形状的输出
    - 测试无掩码情况的处理
    - _Requirements: 3.2, 3.4_

- [x] 5. 检查点 - 模型扩展验证
  - 使用随机输入测试扩展后的模型
  - 确保模型能够正常前向传播
  - 验证通道拼接逻辑正确
  - 如有问题请向用户询问

- [-] 6. 训练流程实现
  - [x] 6.1 实现配置文件解析
    - 编写load_config函数，从YAML文件读取配置参数
    - 验证必需参数的存在性和有效性
    - 提供默认值处理
    - _Requirements: 4.1_
  
  - [ ]* 6.2 编写属性测试：配置文件解析
    - **Property 13: 配置文件解析**
    - **Validates: Requirements 4.1**
  
  - [x] 6.3 实现训练流程管理器
    - 创建TrainingPipeline类
    - 实现setup_model方法：初始化模型、优化器、学习率调度器
    - 实现train_epoch方法：执行一个epoch的训练
    - **⚠️ 关键：移植原项目的Loss计算逻辑（参考codes/transport），确保训练目标与预训练模型一致（Flow Matching/Rectified Flow，而非简单的Noise Prediction）**
    - 实现检查点保存和加载功能
    - 实现训练损失日志记录
    - _Requirements: 4.2, 4.3, 4.4_
  
  - [ ]* 6.4 编写属性测试：检查点保存和恢复
    - **Property 14: 检查点保存周期性**
    - **Property 15: 训练损失日志记录**
    - **Property 16: 检查点恢复一致性**
    - **Validates: Requirements 4.2, 4.3, 4.4**
  
  - [x] 6.5 创建训练脚本
    - 编写train.py脚本，整合数据加载、模型训练、检查点保存
    - 支持命令行参数指定配置文件
    - 添加训练进度显示（tqdm）
    - **集成torch.amp (Automatic Mixed Precision)以支持BF16/FP16训练，减少显存占用并加速（A100服务器优化）**
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 7. 检查点 - 训练流程验证
  - 使用小规模数据（如10张图）和小型模型配置进行短时间训练测试
  - 验证检查点保存和日志记录功能
  - 测试从检查点恢复训练
  - 确保Loss能够下降
  - 如有问题请向用户询问

- [-] 8. 推理生成与数据导出模块实现
  - [x] 8.1 实现图像生成引擎
    - 创建ImageGenerator类
    - 实现模型加载功能
    - 实现generate方法：接收文本和可选掩码，使用DDPM/DDIM采样生成图像
    - 实现batch_generate方法：批量生成图像
    - 支持设置随机种子
    - _Requirements: 5.1, 5.2, 5.4_
  
  - [ ]* 8.2 编写属性测试：生成引擎
    - **Property 17: 生成引擎输入灵活性**
    - **Property 18: 采样算法输出有效性**
    - **Property 20: 批量生成数量一致性**
    - **Validates: Requirements 5.1, 5.2, 5.4**
  
  - [x] 8.3 实现图像保存与数据导出功能
    - 编写save_generation_result函数，将生成的图像保存为PNG格式
    - 创建结果目录结构（results/{timestamp}_{uuid}/）
    - 保存metadata.json文件记录生成参数
    - 编写create_dataset_manifest函数，生成JSONL格式的清单文件
    - _Requirements: 5.3, 9.1, 9.2_
  
  - [ ]* 8.4 编写属性测试：图像文件保存
    - **Property 19: 图像文件保存**
    - **Property 23: 生成结果保存**
    - **Property 24: 数据集清单格式正确性**
    - **Validates: Requirements 5.3, 9.1, 9.2**
  
  - [ ]* 8.5 创建推理脚本（可选，Gradio界面已足够）
    - 编写inference.py脚本，支持从命令行或文件读取输入
    - 支持批量生成模式
    - 显示生成进度
    - **注：此任务已标记为可选，因为Task 11的Gradio界面可完全替代命令行推理功能，建议优先完成训练代码**
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 9. 离线质量评估模块实现
  - [x] 9.1 实现质量评估脚本
    - 创建evaluate.py脚本
    - 实现QualityEvaluator类
    - 实现calculate_psnr方法
    - 实现calculate_ssim方法（使用skimage或torchmetrics）
    - 实现evaluate_batch方法：批量评估整个目录
    - 实现save_results方法：保存评估结果到JSON文件
    - _Requirements: 9.2, 9.3, 9.4_
  
  - [ ]* 9.2 编写属性测试：质量指标计算
    - **Property 21: 质量指标计算与保存（离线）**
    - **Validates: Requirements 9.2, 9.3**
  
  - [ ]* 9.3 编写单元测试：PSNR和SSIM计算
    - 使用已知输入输出对验证指标计算正确性
    - 测试边缘情况（相同图像、完全不同图像）
    - _Requirements: 9.2_

- [ ] 10. 检查点 - 推理和评估验证
  - 使用训练好的模型生成测试图像
  - 验证质量评估功能
  - 确保所有输出文件正确保存
  - 如有问题请向用户询问

- [-] 11. Gradio演示界面实现
  - [x] 11.1 创建Gradio应用主文件
    - 创建src/app/demo.py文件
    - 实现generate_image函数：调用ImageGenerator生成图像
    - 处理文件上传、参数设置、错误处理
    - 返回生成图像和状态信息
    - _Requirements: 7.1, 7.2, 7.4_
  
  - [x] 11.2 实现Gradio界面布局
    - 使用gr.Blocks创建界面
    - 定义输入组件：gr.Image（参考图像，可选）、gr.Image（掩码）、gr.Textbox（文本描述）
    - 定义参数组件：gr.Slider（采样步数、引导强度）、gr.Number（随机种子）
    - 定义输出组件：gr.Image（生成结果）、gr.Textbox（状态信息）
    - 添加gr.Examples组件提供示例输入
    - _Requirements: 7.1, 7.3, 8.1, 8.4_
  
  - [x] 11.3 实现事件绑定和错误处理
    - 绑定生成按钮点击事件到generate_image函数
    - 使用gr.Error()处理和显示错误信息
    - 添加进度提示（可选，使用gr.Progress）
    - _Requirements: 7.2, 7.4, 10.1_
  
  - [ ]* 11.4 编写属性测试：Gradio文件格式验证
    - **Property 22: Gradio文件格式验证**
    - **Validates: Requirements 7.2**
  
  - [ ] 11.5 准备示例数据
    - 在examples/目录下准备示例图像和掩码
    - 编写示例文本描述
    - 配置gr.Examples组件
    - _Requirements: 8.4_

- [ ] 12. 最终集成测试与验证
  - [ ]* 12.1 编写集成测试：完整训练流程
    - 测试数据加载 → 模型训练 → 检查点保存的完整流程
    - 使用小规模数据和小型模型配置
    - _Requirements: 1.1, 2.1, 4.2, 4.3_
  
  - [ ]* 12.2 编写集成测试：完整推理流程
    - 测试模型加载 → 图像生成 → 结果保存的完整流程
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [ ] 12.3 Gradio界面人工验收测试
    - 启动Gradio应用：python src/app/demo.py
    - 测试文件上传功能
    - 测试参数设置和生成流程
    - 测试错误处理（无效输入、GPU内存不足等）
    - 测试下载功能（Gradio内置）
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 13. 最终检查点 - 系统完整性验证
  - 运行所有单元测试和属性测试
  - 执行端到端集成测试
  - 验证Gradio界面所有功能
  - 生成测试覆盖率报告（可选）
  - 如有问题请向用户询问

- [-] 14. 下游任务评估与文档准备
  - [x] 14.1 执行下游分类实验（证明增广价值）
    - 编写train_classifier.py脚本，使用ResNet-50作为分类器
    - 实验1：仅使用原始数据训练分类器，记录准确率
    - 实验2：使用"原始+增广"数据训练分类器，记录准确率
    - 对比两次实验结果，计算准确率提升（目标：2%+提升）
    - 保存实验结果到results/downstream_evaluation.json
    - _Requirements: 9.4_
  
  - [x] 14.2 编写项目文档
    - 编写README.md：项目介绍、安装说明、使用示例
    - 编写训练指南：如何准备数据、配置参数、启动训练
    - 编写推理指南：如何使用命令行脚本和Gradio界面
    - 编写评估指南：如何使用离线评估脚本和下游分类实验
    - _Requirements: 所有需求_
  
  - [ ] 14.3 准备答辩材料
    - 准备答辩PPT素材：系统架构图、生成结果对比、质量指标图表
    - 整理下游分类实验结果图表（对比柱状图）
    - 准备演示视频或截图（Gradio界面使用流程）
    - _Requirements: 所有需求_

## Notes

- 标记为`*`的任务是可选的测试任务，可以跳过以加快MVP开发
- 每个任务都引用了具体的需求编号，确保可追溯性
- 检查点任务确保渐进式验证，及早发现问题
- 属性测试使用Hypothesis库，每个测试运行100次迭代
- 单元测试使用pytest框架
- 所有代码应遵循PEP 8风格规范
- 使用Python 3.10+和PyTorch 2.0+

## A100 服务器优化建议

由于你拥有 A100 服务器使用权，以下优化将显著提升训练效率：

1. **Flash Attention 2**：在 requirements.txt 中添加 `flash-attn`，保留原项目中的 Flash Attention 代码，可将训练速度提升 2-3 倍
2. **混合精度训练**：使用 `torch.amp` 支持 BF16 训练（A100 对 BF16 支持最佳），减少显存占用并加速
3. **批次大小**：A100 拥有 40GB/80GB 显存，可以使用更大的批次大小（建议从 32 开始尝试）
4. **数据加载**：使用多进程数据加载（`num_workers=4-8`）充分利用 CPU 资源

## 推荐数据集（糖尿病视网膜病变）

根据题目要求（图像 + 分割掩码 + 诊断标签/文本），以下两个数据集最适合本项目：

### 推荐一：FGADR (Fine-Grained Annotated Diabetic Retinopathy) ⭐ 最推荐

**最适合本毕设**，直接提供像素级病灶分割掩码。

- **图像**：1,842 张高清眼底图（Seg-set）
- **结构信息（Mask）**：包含微动脉瘤(MA)、出血(HE)、硬性渗出(EX)、软性渗出(SE) 的像素级掩码，完美适配 Task 4
- **语义信息（Text）**：包含 DR 分级标签（0-4级），通过规则转为文本描述
- **下载链接**：
  - GitHub: https://github.com/csyizhou/FGADR-2842-Dataset
  - 项目主页: https://csyizhou.github.io/FGADR/
- **使用方式**：将 4 种病灶的掩码合并为一个 Channel（或保留多通道），作为 Segmentation Mask 输入

### 推荐二：IDRiD (Indian Diabetic Retinopathy Image Dataset)

数据质量极高，常用于顶级竞赛，掩码非常精细。

- **图像**：516 张训练/测试图像
- **结构信息（Mask）**：提供病灶（MA, HE, EX, SE）及视盘 (Optic Disc) 的分割掩码
- **语义信息**：提供 DR 分级和 DME 分级标签
- **下载链接**：
  - IEEE Dataport: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
  - Grand Challenge: https://idrid.grand-challenge.org/
- **优势**：同时包含"视盘"的分割，如果文本描述涉及解剖结构（如"视盘清晰"），这个数据集更好

### 标签转文本示例

在 Task 2.5 中实现的 Label-to-Caption 映射示例：

```python
DR_GRADE_TO_TEXT = {
    0: "No diabetic retinopathy",
    1: "Mild non-proliferative diabetic retinopathy",
    2: "Moderate non-proliferative diabetic retinopathy",
    3: "Severe non-proliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy"
}
```

## 简化说明

相比原计划，本版本做了以下简化：

1. **Web界面**：使用Gradio替代Flask全栈开发，代码量从500-800行降至30-50行
2. **掩码处理**：简化为直接的分辨率调整和通道拼接，无需复杂的One-Hot编码
3. **质量评估**：改为离线脚本，不集成到实时生成流程中
4. **数据导出**：简化为基本的文件保存和清单生成，移除复杂的数据集格式转换
5. **测试覆盖**：降低覆盖率要求（70%），专注核心功能测试
6. **属性测试**：保留核心属性测试，其他标记为可选

这些简化让你能够专注于毕设的核心内容：医学影像增广算法，而不是Web开发细节。
