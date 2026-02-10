# 服务器环境配置指南

## 环境限制

本项目针对特定服务器环境进行了优化，请确保满足以下环境要求：

- **服务器路径限制**：仅能在 `/home/Backup/maziheng` 目录下操作
- **GPU**：NVIDIA A100 40GB
- **Python**：3.10+
- **CUDA**：11.8+
- **操作系统**：Linux

## 快速启动

### 1. 克隆项目

```bash
cd /home/Backup/maziheng
git clone <repo_url> medical-image-augmentation-system
cd medical-image-augmentation-system
```

### 2. 检查依赖

在安装任何依赖之前，先检查服务器上已有的包，避免重复安装和版本冲突：

```bash
python scripts/check_dependencies.py
```

该脚本会输出：
- 已安装的关键包及其版本
- 缺失的包列表
- PyTorch 和 CUDA 版本兼容性检查
- 相关警告信息

### 3. 安装依赖

根据依赖检查报告，手动安装缺失的包：

```bash
# 安装常规依赖
pip install -r requirements.txt

# 手动安装 flash-attn（可选，但强烈推荐用于性能优化）
# 注意：flash-attn 需要编译，确保已安装 CUDA 开发工具
pip install flash-attn --no-build-isolation
```

**重要提示**：
- 手动安装依赖可以避免版本冲突
- 如果某些包已经安装，可以跳过
- flash-attn 安装可能需要 5-10 分钟

### 4. 清理项目结构

如果项目中存在旧的 `codes/` 目录（来自 RetinaLogos 项目），需要安全清理：

```bash
# 步骤 1：安全重命名（不直接删除）
mv codes codes_backup

# 步骤 2：运行冒烟测试，验证 src/ 的独立性
python scripts/smoke_test.py

# 步骤 3：如果测试通过，删除备份
rm -rf codes_backup

# 步骤 4：如果测试失败，回滚操作
python scripts/rollback.py
```

**注意**：如果项目中没有 `codes/` 目录，可以跳过此步骤。

### 5. 验证配置

在开始训练前，验证配置文件的合理性：

```bash
# 检查路径配置
python scripts/check_paths.py

# 验证训练配置（批次大小、显存估算等）
python scripts/validate_config.py
```

根据验证报告调整配置文件：
- 确保所有路径使用相对路径或 `/home/Backup/maziheng` 前缀
- 确保 Effective Batch Size 合理（建议 >= 32）
- 确保显存需求在 A100 40GB 范围内

### 6. 开始训练

配置验证通过后，即可开始训练：

```bash
# 使用默认配置
python train.py --config configs/train_config.yaml

# 或使用快速训练配置（用于测试）
python train.py --config configs/train_config_fast.yaml
```

训练过程中的输出：
- 检查点保存在 `./checkpoints/` 目录
- 日志保存在 `./logs/` 目录
- TensorBoard 日志可通过 `tensorboard --logdir logs/` 查看

## 常见问题

### Q1: flash-attn 安装失败？

**症状**：
```
ERROR: Failed building wheel for flash-attn
```

**解决方案**：

1. 检查 CUDA 版本：
```bash
nvcc --version
```

2. 确保 CUDA 版本 >= 11.8

3. 检查 gcc 版本：
```bash
gcc --version
```

4. 确保 gcc 版本 >= 7.0

5. 如果环境满足要求，尝试重新安装：
```bash
pip install flash-attn --no-build-isolation --verbose
```

6. 如果仍然失败，可以跳过 flash-attn（性能会降低约 20-30%）：
```bash
# 在 src/models/nexdit_mask.py 中注释掉 flash-attn 相关代码
# 使用标准的 PyTorch attention 实现
```

### Q2: 显存不足（OOM）错误？

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：

1. 减小 batch_size：
```yaml
# configs/train_config.yaml
training:
  batch_size: 1  # 从 2 减小到 1
  gradient_accumulation_steps: 32  # 相应增加以保持 Effective Batch Size
```

2. 减小图像尺寸：
```yaml
# configs/train_config.yaml
data:
  image_size: 512  # 从 1024 减小到 512
```

3. 减小模型维度：
```yaml
# configs/train_config.yaml
model:
  dim: 2048  # 从 4096 减小到 2048
  n_layers: 16  # 从 32 减小到 16
```

4. 启用梯度检查点（gradient checkpointing）：
```yaml
# configs/train_config.yaml
training:
  gradient_checkpointing: true
```

**注意**：调整配置后，重新运行 `python scripts/validate_config.py` 验证。

### Q3: 路径错误？

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory: '/some/absolute/path'
```

**解决方案**：

1. 运行路径检查脚本：
```bash
python scripts/check_paths.py
```

2. 根据报告修改配置文件，确保所有路径使用相对路径：
```yaml
# 错误示例
data:
  train_data: "/home/user/data/train.jsonl"  # 绝对路径

# 正确示例
data:
  train_data: "data/train.jsonl"  # 相对路径
```

3. 如果必须使用绝对路径，确保使用服务器限制路径前缀：
```yaml
data:
  train_data: "/home/Backup/maziheng/medical-image-augmentation-system/data/train.jsonl"
```

### Q4: 依赖版本冲突？

**症状**：
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**解决方案**：

1. 运行依赖检查脚本：
```bash
python scripts/check_dependencies.py
```

2. 手动卸载冲突的包：
```bash
pip uninstall <conflicting_package>
```

3. 按照 requirements.txt 重新安装：
```bash
pip install <package>==<version>
```

4. 如果冲突无法解决，考虑使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Q5: 数据加载速度慢？

**症状**：训练时数据加载成为瓶颈

**解决方案**：

1. 增加数据加载线程数：
```yaml
# configs/train_config.yaml
training:
  num_workers: 8  # 根据 CPU 核心数调整
```

2. 启用数据预加载：
```yaml
# configs/train_config.yaml
data:
  prefetch_factor: 2
  persistent_workers: true
```

3. 检查数据存储位置，确保在高速存储上（SSD 优于 HDD）

### Q6: 训练不稳定或不收敛？

**症状**：Loss 震荡或不下降

**解决方案**：

1. 检查 Effective Batch Size：
```bash
python scripts/validate_config.py
```

2. 确保 Effective Batch Size >= 32：
```yaml
# configs/train_config.yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 16  # 2 × 16 = 32
```

3. 调整学习率：
```yaml
# configs/train_config.yaml
training:
  learning_rate: 1.0e-6  # 如果不稳定，可以减小到 5.0e-7
```

4. 启用梯度裁剪：
```yaml
# configs/train_config.yaml
training:
  max_grad_norm: 1.0
```

## 配置说明

### Effective Batch Size 计算

Effective Batch Size = batch_size × gradient_accumulation_steps

**示例**：
- batch_size = 2, gradient_accumulation_steps = 16
- Effective Batch Size = 2 × 16 = 32

**建议**：
- 对于 1024×1024 图像：Effective Batch Size >= 32
- 对于 512×512 图像：Effective Batch Size >= 64

### A100 优化配置

当前配置已针对 A100 40GB 进行优化：

```yaml
training:
  batch_size: 2  # 针对 1024×1024 图像
  gradient_accumulation_steps: 16
  mixed_precision: "bf16"  # A100 支持 BF16，比 FP16 更稳定
  num_workers: 8  # 充分利用 CPU
```

### 路径配置原则

1. **优先使用相对路径**：
   - 数据路径：`data/train.jsonl`
   - 输出路径：`./checkpoints`, `./logs`

2. **必要时使用绝对路径**：
   - 必须以 `/home/Backup/maziheng` 开头
   - 示例：`/home/Backup/maziheng/medical-image-augmentation-system/data/train.jsonl`

3. **避免的路径**：
   - 用户主目录：`~/data` ❌
   - 其他用户目录：`/home/other_user/` ❌
   - 系统目录：`/tmp`, `/var` ❌

## 性能优化建议

### 1. 数据预处理

- 如果数据集已经过预处理，禁用去噪功能：
```yaml
data:
  enable_denoising: false
```

### 2. 混合精度训练

- A100 支持 BF16，比 FP16 更稳定：
```yaml
training:
  mixed_precision: "bf16"
```

### 3. 编译优化

- 使用 PyTorch 2.0+ 的编译功能：
```python
# 在 train.py 中添加
model = torch.compile(model)
```

### 4. Flash Attention

- 强烈推荐安装 flash-attn，可提升 20-30% 性能：
```bash
pip install flash-attn --no-build-isolation
```

## 故障排查流程

如果遇到问题，按以下顺序排查：

1. **检查依赖**：
   ```bash
   python scripts/check_dependencies.py
   ```

2. **检查路径**：
   ```bash
   python scripts/check_paths.py
   ```

3. **验证配置**：
   ```bash
   python scripts/validate_config.py
   ```

4. **运行冒烟测试**：
   ```bash
   python scripts/smoke_test.py
   ```

5. **查看日志**：
   ```bash
   tail -f logs/cleanup.log
   tail -f logs/train.log
   ```

6. **如果需要回滚**：
   ```bash
   python scripts/rollback.py
   ```

## 联系支持

如果以上方法都无法解决问题，请：

1. 收集错误信息：
   - 完整的错误堆栈
   - 配置文件内容
   - 依赖检查报告

2. 检查项目文档：
   - README.md
   - SETUP.md
   - CLEANUP_CHECKLIST.md

3. 查看相关日志文件：
   - logs/cleanup.log
   - logs/train.log

## 附录

### 推荐的开发工作流

1. 克隆项目并检查依赖
2. 安装必要的包
3. 清理项目结构（如果需要）
4. 验证配置
5. 运行快速训练测试（使用 train_config_fast.yaml）
6. 开始完整训练

### 常用命令速查

```bash
# 依赖管理
python scripts/check_dependencies.py

# 路径检查
python scripts/check_paths.py

# 配置验证
python scripts/validate_config.py

# 冒烟测试
python scripts/smoke_test.py

# 回滚操作
python scripts/rollback.py

# 开始训练
python train.py --config configs/train_config.yaml

# 查看 TensorBoard
tensorboard --logdir logs/

# 运行测试
python -m pytest tests/
```
