# 项目清理操作检查清单

本文档提供项目清理和服务器适配的完整检查清单，确保每个步骤都正确执行。

## 📋 目录

- [清理前准备](#清理前准备)
- [清理步骤检查清单](#清理步骤检查清单)
- [验证方法](#验证方法)
- [回滚指南](#回滚指南)
- [常见问题和解决方案](#常见问题和解决方案)

## 清理前准备

### ✅ 环境检查

在开始清理前，确保满足以下条件：

- [ ] Python 3.10+ 已安装
- [ ] CUDA 11.8+ 已安装（如果使用 GPU）
- [ ] 有足够的磁盘空间（至少 10GB）
- [ ] 有项目的备份或版本控制（Git）
- [ ] 了解服务器路径限制（如适用）

### ✅ 工具脚本检查

确保所有工具脚本已创建：

- [ ] `scripts/check_dependencies.py` 存在
- [ ] `scripts/check_paths.py` 存在
- [ ] `scripts/validate_config.py` 存在
- [ ] `scripts/smoke_test.py` 存在
- [ ] `scripts/rollback.py` 存在

验证方法：
```bash
ls -la scripts/
```

### ✅ 日志目录检查

确保日志目录存在：

- [ ] `logs/` 目录存在
- [ ] 有写入权限

验证方法：
```bash
mkdir -p logs
touch logs/test.log && rm logs/test.log
```

## 清理步骤检查清单

### 步骤 1: 依赖检查 ⏱️ 5 分钟

#### 任务

- [ ] 运行依赖检查脚本
- [ ] 查看已安装的包列表
- [ ] 记录缺失的包
- [ ] 检查 PyTorch 和 CUDA 兼容性

#### 执行命令

```bash
python scripts/check_dependencies.py
```

#### 验证方法

检查输出中是否包含：
- ✅ 已安装包数量
- ✅ 关键依赖状态（torch, numpy, pillow 等）
- ✅ CUDA 兼容性检查结果

#### 预期输出

```
=== 依赖检查报告 ===
已安装包数量: 156
✓ torch: 2.0.1
✓ torchvision: 0.15.2
✓ numpy: 1.24.3
✓ pillow: 10.0.0
✓ PyTorch 和 CUDA 版本兼容
```

#### 如果失败

- 检查 Python 环境是否正确
- 确保 pip 可用
- 查看错误信息并修复

---

### 步骤 2: 安装缺失依赖 ⏱️ 10-15 分钟

#### 任务

- [ ] 根据依赖检查报告安装缺失的包
- [ ] 手动安装 flash-attn（可选）
- [ ] 验证安装结果

#### 执行命令

```bash
# 安装常规依赖
pip install -r requirements.txt

# 手动安装 flash-attn（可选）
pip install flash-attn --no-build-isolation
```

#### 验证方法

重新运行依赖检查：
```bash
python scripts/check_dependencies.py
```

#### 预期结果

- ✅ 所有关键依赖已安装
- ✅ 无缺失包警告

#### 如果失败

- 检查网络连接
- 查看 pip 错误信息
- 对于 flash-attn，检查 CUDA 和 gcc 版本

---

### 步骤 3: 路径检查 ⏱️ 5 分钟

#### 任务

- [ ] 运行路径检查脚本
- [ ] 检查所有配置文件中的路径
- [ ] 记录不符合规范的路径

#### 执行命令

```bash
python scripts/check_paths.py
```

#### 验证方法

检查输出中是否有路径问题：
- ✅ 所有路径使用相对路径或服务器限制路径
- ✅ 无硬编码绝对路径
- ✅ 无用户主目录路径（~/）

#### 预期输出

```
=== 路径检查报告 ===
检查文件: configs/train_config.yaml
✓ 所有路径检查通过

检查文件: configs/inference_config.yaml
✓ 所有路径检查通过

总结: 所有路径配置正确
```

#### 如果失败

- 根据报告修改配置文件
- 将绝对路径改为相对路径
- 重新运行检查

---

### 步骤 4: 配置验证 ⏱️ 2 分钟

#### 任务

- [ ] 运行配置验证脚本
- [ ] 检查批次配置
- [ ] 检查显存估算
- [ ] 记录配置建议

#### 执行命令

```bash
python scripts/validate_config.py
```

#### 验证方法

检查输出中的配置状态：
- ✅ Effective Batch Size >= 16（建议 >= 32）
- ✅ 显存需求在 GPU 范围内
- ✅ 无配置警告

#### 预期输出

```
=== 配置验证报告 ===
✓ 批次配置: Effective Batch Size: 32 (合理)
估算显存需求: 28.45 GB
✓ 显存需求在 A100 40GB 范围内
```

#### 如果失败

- 根据建议调整配置
- 减小 batch_size 或 image_size
- 增加 gradient_accumulation_steps

---

### 步骤 5: 安全重命名 codes/ 目录 ⏱️ 1 分钟

#### 任务

- [ ] 检查 codes/ 目录是否存在
- [ ] 重命名 codes/ 为 codes_backup/
- [ ] 记录操作到日志

#### 执行命令

```bash
# 检查 codes/ 是否存在
if [ -d "codes" ]; then
    # 重命名
    mv codes codes_backup
    
    # 记录操作
    echo "$(date): Renamed codes to codes_backup" >> logs/cleanup.log
    
    echo "✓ codes/ 已重命名为 codes_backup/"
else
    echo "⚠️ codes/ 目录不存在，跳过此步骤"
fi
```

#### 验证方法

```bash
# 检查目录状态
ls -la | grep codes
```

#### 预期结果

- ✅ codes_backup/ 存在
- ✅ codes/ 不存在
- ✅ 操作已记录到 logs/cleanup.log

#### 如果失败

- 检查文件权限
- 检查是否有进程占用 codes/ 目录
- 手动重命名

---

### 步骤 6: 运行冒烟测试 ⏱️ 2 分钟

#### 任务

- [ ] 运行冒烟测试脚本
- [ ] 验证数据加载功能
- [ ] 验证模型初始化
- [ ] 验证前向传播

#### 执行命令

```bash
python scripts/smoke_test.py
```

#### 验证方法

检查所有测试是否通过：
- ✅ 数据加载测试通过
- ✅ 模型初始化测试通过
- ✅ 前向传播测试通过

#### 预期输出

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

#### 如果失败

**⚠️ 重要：如果测试失败，立即执行回滚！**

```bash
python scripts/rollback.py
```

然后分析失败原因：
- 检查是否有隐式依赖 codes/
- 检查导入路径是否正确
- 查看错误堆栈信息

---

### 步骤 7: 删除 codes_backup/ 目录 ⏱️ 1 分钟

#### 任务

- [ ] 确认冒烟测试已通过
- [ ] 删除 codes_backup/ 目录
- [ ] 记录操作到日志

#### 执行命令

```bash
# 仅在冒烟测试通过后执行
if [ -d "codes_backup" ]; then
    rm -rf codes_backup
    echo "$(date): Deleted codes_backup" >> logs/cleanup.log
    echo "✓ codes_backup/ 已删除"
else
    echo "⚠️ codes_backup/ 不存在"
fi
```

#### 验证方法

```bash
# 检查目录是否已删除
ls -la | grep codes
```

#### 预期结果

- ✅ codes_backup/ 不存在
- ✅ 操作已记录到 logs/cleanup.log

#### 如果失败

- 检查文件权限
- 检查是否有进程占用目录
- 手动删除

---

### 步骤 8: 最终验证 ⏱️ 5-10 分钟

#### 任务

- [ ] 重新运行冒烟测试
- [ ] 运行单元测试（可选）
- [ ] 验证配置文件
- [ ] 检查文档完整性

#### 执行命令

```bash
# 冒烟测试
python scripts/smoke_test.py

# 单元测试（可选）
python -m pytest tests/

# 配置验证
python scripts/validate_config.py

# 路径检查
python scripts/check_paths.py
```

#### 验证方法

检查所有验证是否通过：
- ✅ 冒烟测试通过
- ✅ 单元测试通过（如果运行）
- ✅ 配置验证无警告
- ✅ 路径检查通过

#### 预期结果

所有验证脚本都返回成功状态。

#### 如果失败

- 根据具体错误信息修复
- 查看日志文件
- 参考常见问题章节

---

## 验证方法

### 完整性验证

运行以下命令验证清理操作的完整性：

```bash
# 1. 检查目录结构
echo "=== 检查目录结构 ==="
ls -la | grep -E "src|scripts|tests|configs|logs"

# 2. 检查 codes/ 是否已删除
echo "=== 检查 codes/ 目录 ==="
if [ -d "codes" ] || [ -d "codes_backup" ]; then
    echo "⚠️ codes/ 或 codes_backup/ 仍然存在"
else
    echo "✓ codes/ 已清理"
fi

# 3. 运行所有验证脚本
echo "=== 运行验证脚本 ==="
python scripts/check_dependencies.py
python scripts/check_paths.py
python scripts/validate_config.py
python scripts/smoke_test.py

# 4. 检查日志
echo "=== 检查清理日志 ==="
cat logs/cleanup.log
```

### 功能验证

验证核心功能是否正常：

```bash
# 1. 数据加载
python -c "from src.data.dataset import ImageDataset; print('✓ 数据模块正常')"

# 2. 模型加载
python -c "from src.models.nexdit_mask import NextDiT; print('✓ 模型模块正常')"

# 3. 训练配置
python -c "from src.training.config import TrainingConfig; print('✓ 训练模块正常')"

# 4. 推理模块
python -c "from src.inference.generator import ImageGenerator; print('✓ 推理模块正常')"
```

### 配置验证

验证所有配置文件：

```bash
# 检查配置文件语法
python -c "import yaml; yaml.safe_load(open('configs/train_config.yaml')); print('✓ train_config.yaml 语法正确')"
python -c "import yaml; yaml.safe_load(open('configs/inference_config.yaml')); print('✓ inference_config.yaml 语法正确')"
```

## 回滚指南

### 何时需要回滚

在以下情况下需要执行回滚：

- ❌ 冒烟测试失败
- ❌ 发现隐式依赖 codes/
- ❌ 项目无法正常运行
- ❌ 误删重要文件

### 回滚步骤

#### 1. 立即停止操作

如果发现问题，立即停止后续操作。

#### 2. 运行回滚脚本

```bash
python scripts/rollback.py
```

#### 3. 验证回滚结果

```bash
# 检查 codes/ 是否恢复
ls -la | grep codes

# 运行冒烟测试
python scripts/smoke_test.py
```

#### 4. 分析失败原因

查看日志文件：
```bash
cat logs/cleanup.log
```

检查错误信息：
- 导入错误：可能有隐式依赖
- 文件未找到：路径配置问题
- 模块错误：依赖安装问题

#### 5. 修复问题

根据分析结果修复问题：

**导入错误**：
```bash
# 检查导入路径
grep -r "from codes" src/
grep -r "import codes" src/
```

**依赖问题**：
```bash
# 重新检查依赖
python scripts/check_dependencies.py
```

**路径问题**：
```bash
# 重新检查路径
python scripts/check_paths.py
```

#### 6. 重新尝试清理

修复问题后，从步骤 5 重新开始清理流程。

### 手动回滚

如果回滚脚本失败，手动执行：

```bash
# 1. 恢复 codes/ 目录
if [ -d "codes_backup" ]; then
    mv codes_backup codes
    echo "✓ codes/ 已恢复"
fi

# 2. 验证恢复
python scripts/smoke_test.py

# 3. 记录回滚操作
echo "$(date): Manual rollback completed" >> logs/cleanup.log
```

## 常见问题和解决方案

### Q1: 冒烟测试失败 - 数据加载错误

**症状**：
```
✗ 数据加载测试失败: No module named 'codes.example_data'
```

**原因**：代码中有隐式依赖 codes/ 目录

**解决方案**：
1. 回滚操作：`python scripts/rollback.py`
2. 搜索依赖：`grep -r "codes\." src/`
3. 修改导入路径，将 `codes.` 改为 `src.`
4. 重新测试

---

### Q2: 冒烟测试失败 - 模型初始化错误

**症状**：
```
✗ 模型初始化测试失败: No module named 'codes.models'
```

**原因**：模型代码依赖 codes/ 中的模块

**解决方案**：
1. 回滚操作
2. 检查 src/models/ 中的导入
3. 确保所有模型代码已迁移到 src/
4. 更新导入路径

---

### Q3: 路径检查失败 - 绝对路径

**症状**：
```
⚠️ 发现路径问题：
  configs/train_config.yaml:
    - /home/user/data/train.jsonl
```

**原因**：配置文件中使用了硬编码绝对路径

**解决方案**：
1. 编辑配置文件：`vim configs/train_config.yaml`
2. 将绝对路径改为相对路径：
   ```yaml
   # 修改前
   train_data: "/home/user/data/train.jsonl"
   
   # 修改后
   train_data: "data/train.jsonl"
   ```
3. 重新运行检查：`python scripts/check_paths.py`

---

### Q4: 配置验证失败 - Effective Batch Size 过小

**症状**：
```
✗ 批次配置: Effective Batch Size (8) 过小，可能导致训练不稳定
```

**原因**：batch_size 和 gradient_accumulation_steps 配置不合理

**解决方案**：
1. 编辑配置文件：`vim configs/train_config.yaml`
2. 调整配置：
   ```yaml
   training:
     batch_size: 2
     gradient_accumulation_steps: 16  # 确保 2 × 16 = 32
   ```
3. 重新验证：`python scripts/validate_config.py`

---

### Q5: 显存不足（OOM）

**症状**：
```
RuntimeError: CUDA out of memory
```

**原因**：batch_size 或 image_size 过大

**解决方案**：
1. 减小 batch_size：
   ```yaml
   training:
     batch_size: 1  # 从 2 减小到 1
     gradient_accumulation_steps: 32  # 相应增加
   ```
2. 或减小 image_size：
   ```yaml
   data:
     image_size: 512  # 从 1024 减小到 512
   ```
3. 重新验证：`python scripts/validate_config.py`

---

### Q6: flash-attn 安装失败

**症状**：
```
ERROR: Failed building wheel for flash-attn
```

**原因**：CUDA 或 gcc 版本不兼容

**解决方案**：
1. 检查 CUDA 版本：`nvcc --version`（需要 >= 11.8）
2. 检查 gcc 版本：`gcc --version`（需要 >= 7.0）
3. 如果版本满足，重试：
   ```bash
   pip install flash-attn --no-build-isolation --verbose
   ```
4. 如果仍然失败，可以跳过（性能会降低 20-30%）

---

### Q7: 依赖版本冲突

**症状**：
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**原因**：已安装的包版本与 requirements.txt 冲突

**解决方案**：
1. 查看冲突详情
2. 手动卸载冲突的包：
   ```bash
   pip uninstall <conflicting_package>
   ```
3. 重新安装：
   ```bash
   pip install <package>==<version>
   ```
4. 或使用虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

### Q8: 回滚脚本失败

**症状**：
```
没有找到清理操作日志
```

**原因**：logs/cleanup.log 不存在或为空

**解决方案**：
手动回滚：
```bash
# 恢复 codes/ 目录
if [ -d "codes_backup" ]; then
    mv codes_backup codes
fi

# 验证恢复
python scripts/smoke_test.py
```

---

### Q9: 单元测试失败

**症状**：
```
FAILED tests/test_preprocessing.py::test_image_resize
```

**原因**：测试代码或实现代码有问题

**解决方案**：
1. 查看详细错误信息：
   ```bash
   python -m pytest tests/test_preprocessing.py -v
   ```
2. 根据错误信息修复代码
3. 重新运行测试

---

### Q10: 文档缺失

**症状**：找不到 SERVER_SETUP.md 或其他文档

**原因**：文档未创建或被误删

**解决方案**：
1. 检查文档是否存在：
   ```bash
   ls -la *.md
   ```
2. 如果缺失，从 Git 恢复或重新创建
3. 参考本检查清单重新执行相应步骤

---

## 清理完成检查

完成所有步骤后，进行最终检查：

### ✅ 目录结构检查

- [ ] codes/ 目录已删除
- [ ] codes_backup/ 目录已删除
- [ ] src/ 目录完整
- [ ] scripts/ 目录包含所有工具脚本
- [ ] tests/ 目录包含测试文件
- [ ] configs/ 目录包含配置文件
- [ ] logs/ 目录存在且包含 cleanup.log

### ✅ 功能检查

- [ ] 冒烟测试全部通过
- [ ] 单元测试全部通过（如果运行）
- [ ] 数据加载功能正常
- [ ] 模型初始化功能正常
- [ ] 配置文件验证通过

### ✅ 配置检查

- [ ] 所有路径使用相对路径或服务器限制路径
- [ ] Effective Batch Size >= 16（建议 >= 32）
- [ ] 显存需求在 GPU 范围内
- [ ] 混合精度配置正确（BF16 for A100）

### ✅ 依赖检查

- [ ] 所有关键依赖已安装
- [ ] PyTorch 和 CUDA 版本兼容
- [ ] flash-attn 已安装（可选）
- [ ] 无依赖冲突警告

### ✅ 文档检查

- [ ] README.md 已更新
- [ ] SETUP.md 已更新
- [ ] SERVER_SETUP.md 已创建
- [ ] CLEANUP_CHECKLIST.md 已创建（本文件）
- [ ] 所有文档内容完整

## 成功标志

当以下所有条件满足时，清理操作成功完成：

✅ codes/ 目录已安全删除  
✅ 所有依赖已检查并手动安装  
✅ 所有配置文件使用相对路径  
✅ A100 优化配置已应用  
✅ 冒烟测试全部通过  
✅ 配置验证无警告  
✅ 文档已更新完整  
✅ 简化的测试策略已实施  

## 下一步

清理完成后，可以：

1. **开始训练**：
   ```bash
   python train.py --config configs/train_config.yaml
   ```

2. **运行快速训练测试**：
   ```bash
   python train.py --config configs/train_config_fast.yaml
   ```

3. **启动 Gradio 演示**：
   ```bash
   python src/app/demo.py --checkpoint checkpoints/best_model.pth
   ```

4. **进行下游评估**：
   ```bash
   python evaluate.py --generated results/ --reference data/reference/
   ```

## 参考文档

- **快速启动**：[README.md](README.md)
- **详细安装**：[SETUP.md](SETUP.md)
- **服务器配置**：[SERVER_SETUP.md](SERVER_SETUP.md)

## 联系支持

如果遇到本文档未涵盖的问题：

1. 查看日志文件：`logs/cleanup.log`
2. 运行诊断脚本收集信息
3. 查看相关文档的常见问题章节
4. 提交 Issue 或联系项目维护者

---

**最后更新**：2024-01-01  
**版本**：1.0
