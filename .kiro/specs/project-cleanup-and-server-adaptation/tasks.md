# Implementation Plan: Project Cleanup and Server Adaptation

## Overview

本实现计划针对单人开发、A100 服务器环境以及严格的路径限制，提供项目清理和服务器适配的详细任务列表。核心目标是安全清理冗余代码、手动管理依赖、适配服务器路径、优化 A100 配置，并简化测试策略。

**预计总时间**：25-45 分钟

## Tasks

- [x] 1. 创建工具脚本目录和基础文件
  - 创建 scripts/ 目录
  - 创建空的脚本文件：check_dependencies.py, check_paths.py, validate_config.py, smoke_test.py, rollback.py
  - 创建 logs/ 目录（如果不存在）
  - 创建 tests/ 目录（如果不存在）
  - _Requirements: 1.5, 2.4, 7.1_

- [x] 2. 实现依赖检查脚本
  - 编写 check_dependencies.py
  - 实现 check_installed_packages() 函数：使用 pkg_resources 获取已安装包
  - 实现 check_pytorch_cuda_compatibility() 函数：检查 PyTorch 和 CUDA 版本
  - 实现 list_missing_packages() 函数：对比 requirements.txt 找出缺失包
  - 实现 main() 函数：生成依赖检查报告
  - _Requirements: 2.4, 8.1, 8.2, 8.3, 8.4_

- [x] 3. 实现路径检查脚本
  - 编写 check_paths.py
  - 实现 check_yaml_paths() 函数：解析 YAML 文件并检查路径
  - 实现 is_valid_path() 函数：验证路径是否符合服务器限制
  - 实现 convert_to_relative_path() 函数：提供路径转换建议
  - 实现 main() 函数：检查所有配置文件并生成报告
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 4. 实现配置验证脚本
  - 编写 validate_config.py
  - 实现 validate_batch_config() 函数：检查 batch_size 和 gradient_accumulation_steps
  - 实现 estimate_gpu_memory() 函数：粗略估算显存需求
  - 实现 main() 函数：生成配置验证报告
  - _Requirements: 4.1, 4.2, 7.2, 7.3, 7.4_

- [x] 5. 实现冒烟测试脚本
  - 编写 smoke_test.py
  - 实现 test_data_loading() 函数：测试数据加载功能
  - 实现 test_model_initialization() 函数：测试模型初始化
  - 实现 test_forward_pass() 函数：测试前向传播
  - 实现 main() 函数：运行所有测试并返回退出码
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. 实现回滚脚本
  - 编写 rollback.py
  - 实现 read_cleanup_log() 函数：读取清理操作日志
  - 实现 rollback_operation() 函数：回滚单个操作
  - 实现 main() 函数：执行完整回滚流程
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 7. 清理和更新 requirements.txt
  - 检查当前 requirements.txt 和 codes/environment_RetinaLogos.yml
  - 手动整理依赖列表，移除重复和不必要的包
  - 添加核心依赖及其版本范围
  - 添加 flash-attn 的手动安装说明（注释形式）
  - 添加依赖分类注释（核心框架、图像处理、数据处理等）
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 8. 检查点 - 依赖验证
  - 运行 python scripts/check_dependencies.py
  - 检查报告中的已安装包和缺失包
  - 根据报告手动安装缺失的包（不要自动安装）
  - 验证 PyTorch 和 CUDA 版本兼容性
  - 如有问题请向用户询问

- [x] 9. 适配配置文件路径
  - 打开 configs/train_config.yaml
  - 检查所有路径字段（data.train_data, data.val_data, output.checkpoint_dir, output.log_dir）
  - 将绝对路径改为相对路径
  - 添加路径说明注释
  - 添加 Effective Batch Size 计算说明注释
  - 重复以上步骤处理 configs/inference_config.yaml
  - _Requirements: 3.1, 3.2, 3.4, 3.5_

- [x] 10. 优化 A100 配置
  - 打开 configs/train_config.yaml
  - 设置 training.batch_size = 2（针对 1024x1024 图像）
  - 设置 training.gradient_accumulation_steps = 16（确保 Effective Batch Size = 32）
  - 设置 training.mixed_precision = "bf16"（A100 优化）
  - 设置 data.num_workers = 8（充分利用 CPU）
  - 添加配置说明注释
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 11. 检查点 - 路径和配置验证
  - 运行 python scripts/check_paths.py
  - 检查报告中的路径问题
  - 根据报告修改配置文件
  - 运行 python scripts/validate_config.py
  - 检查 Effective Batch Size 和显存估算
  - 如有问题请向用户询问

- [~] 12. 安全重命名 codes/ 目录
  - 在项目根目录执行：mv codes codes_backup
  - 记录操作到 logs/cleanup.log：echo "$(date): Renamed codes to codes_backup" >> logs/cleanup.log
  - 确认 codes_backup/ 存在且 codes/ 不存在
  - _Requirements: 1.1, 1.5_

- [~] 13. 运行冒烟测试
  - 运行 python scripts/smoke_test.py
  - 检查测试结果：数据加载、模型初始化、前向传播
  - 如果所有测试通过，继续下一步
  - 如果任何测试失败，运行 python scripts/rollback.py 并分析依赖问题
  - _Requirements: 1.2, 1.3, 1.4_

- [~] 14. 删除 codes_backup/ 目录（仅在测试通过后）
  - 确认冒烟测试已通过
  - 在项目根目录执行：rm -rf codes_backup
  - 记录操作到 logs/cleanup.log：echo "$(date): Deleted codes_backup" >> logs/cleanup.log
  - 确认 codes_backup/ 已删除
  - _Requirements: 1.3_

- [x] 15. 补充噪声处理功能
  - 打开 src/data/preprocessing.py
  - 添加 apply_gaussian_filter() 函数：实现高斯滤波去噪
  - 添加 apply_median_filter() 函数：实现中值滤波去噪
  - 在 preprocess_image() 函数中添加可选的去噪步骤（根据配置）
  - 在 configs/train_config.yaml 中添加 data.enable_denoising 配置项（默认 false）
  - _Requirements: 6.1, 6.2, 6.3, 6.5_

- [x] 16. 创建简化的单元测试
  - 创建 tests/test_preprocessing.py
  - 编写 test_image_resize() 测试：验证图像调整大小
  - 编写 test_mask_resize() 测试：验证掩码调整大小
  - 编写 test_normalization() 测试：验证像素值归一化
  - 创建 tests/test_model.py
  - 编写 test_model_initialization() 测试：验证模型初始化
  - 编写 test_forward_pass() 测试：验证前向传播
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 17. 创建服务器配置文档
  - 创建 SERVER_SETUP.md 文件
  - 编写"环境限制"章节：说明服务器路径限制、GPU、Python、CUDA 版本
  - 编写"快速启动"章节：包含克隆项目、检查依赖、安装依赖、清理结构、验证配置、开始训练的步骤
  - 编写"常见问题"章节：包含 flash-attn 安装、显存不足、路径错误的解决方案
  - 添加命令示例和代码块
  - _Requirements: 10.3, 10.4, 10.5_

- [x] 18. 更新 README.md
  - 添加"项目结构"章节：说明简化后的目录结构
  - 更新"安装说明"章节：引用 SERVER_SETUP.md
  - 添加"简化说明"章节：说明避免过度工程化的原则
  - 更新"快速启动"章节：提供最简化的启动步骤
  - _Requirements: 10.1, 10.4_

- [x] 19. 更新 SETUP.md
  - 更新"依赖安装"章节：强调手动安装而非自动化脚本
  - 添加"依赖检查"章节：说明如何使用 check_dependencies.py
  - 更新"配置文件"章节：说明路径适配和 A100 优化
  - 添加"测试策略"章节：说明简化的测试方法（冒烟测试 + 单元测试）
  - _Requirements: 10.2, 10.4_

- [~] 20. 最终验证
  - 运行完整的测试套件：python -m pytest tests/
  - 或运行核心测试：python tests/test_preprocessing.py && python tests/test_model.py
  - 再次运行冒烟测试：python scripts/smoke_test.py
  - 再次运行配置验证：python scripts/validate_config.py
  - 检查所有文档是否更新完整
  - 如有问题请向用户询问

- [x] 21. 创建清理操作检查清单
  - 创建 CLEANUP_CHECKLIST.md 文件
  - 列出所有清理步骤的检查项
  - 添加每个步骤的验证方法
  - 添加回滚指南
  - 添加常见问题和解决方案
  - _Requirements: 10.5_

## Notes

- 所有脚本都应该有清晰的输出和错误提示
- 不要编写自动化的依赖安装脚本，手动安装更安全
- 不要直接删除 codes/，先重命名并测试
- 不要实现 hypothesis 属性测试，专注冒烟测试和简单单元测试
- 所有路径使用相对路径或 /home/Backup/maziheng 前缀
- Effective Batch Size 必须 >= 16，建议 >= 32
- flash-attn 需要手动编译安装，提供清晰的安装说明
- 配置文件中添加详细的注释说明

## 执行顺序建议

1. **准备阶段**（Tasks 1-7）：创建脚本和清理依赖文件
2. **验证阶段**（Tasks 8-11）：检查依赖和配置
3. **清理阶段**（Tasks 12-14）：安全清理 codes/ 目录
4. **补充阶段**（Tasks 15-16）：补充功能和测试
5. **文档阶段**（Tasks 17-19）：更新文档
6. **验证阶段**（Tasks 20-21）：最终验证和检查清单

## 时间估算

- Tasks 1-7（准备）：10-15 分钟
- Tasks 8-11（验证）：5-10 分钟
- Tasks 12-14（清理）：5 分钟
- Tasks 15-16（补充）：10-15 分钟
- Tasks 17-19（文档）：10-15 分钟
- Tasks 20-21（验证）：5-10 分钟
- **总计**：45-70 分钟

## 成功标志

- ✓ codes/ 目录已安全删除
- ✓ 所有依赖已检查并手动安装
- ✓ 所有配置文件使用相对路径
- ✓ A100 优化配置已应用
- ✓ 冒烟测试全部通过
- ✓ 配置验证无警告
- ✓ 文档已更新完整
- ✓ 简化的测试策略已实施

## 回滚指南

如果在任何步骤遇到问题：

1. 运行回滚脚本：`python scripts/rollback.py`
2. 检查项目结构：`ls -la`
3. 重新运行冒烟测试：`python scripts/smoke_test.py`
4. 分析失败原因并修复
5. 从失败的步骤重新开始
