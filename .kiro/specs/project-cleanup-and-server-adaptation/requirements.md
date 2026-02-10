# Requirements Document: Project Cleanup and Server Adaptation

## Introduction

本文档定义了项目清理和服务器环境适配的需求。针对单人开发、A100服务器环境以及严格的路径限制（仅能在 /home/Backup/maziheng 操作），本规范旨在确保项目结构清晰、依赖管理明确、配置文件适配服务器环境，并避免过度工程化。

## Glossary

- **codes/**: 旧项目（RetinaLogos）的残留代码目录
- **src/**: 本项目的核心代码目录
- **Server_Path**: 服务器限制路径 /home/Backup/maziheng
- **Smoke_Test**: 冒烟测试，用于快速验证核心功能是否正常
- **A100_Optimization**: 针对 A100 GPU 的优化配置
- **Effective_Batch_Size**: 有效批次大小 = batch_size × gradient_accumulation_steps

## Requirements

### Requirement 1: 安全的项目结构清理

**User Story:** 作为开发者，我希望安全地清理冗余代码目录，避免误删导致项目无法运行。

#### Acceptance Criteria

1. WHEN 执行清理操作 THEN THE System SHALL 先将 codes/ 重命名为 codes_backup/
2. WHEN codes/ 被重命名后 THEN THE System SHALL 运行冒烟测试验证 src/ 的独立性
3. WHEN 冒烟测试通过 THEN THE System SHALL 提示用户可以安全删除 codes_backup/
4. WHEN 冒烟测试失败 THEN THE System SHALL 恢复 codes_backup/ 为 codes/ 并报告依赖问题
5. THE System SHALL 记录清理操作日志到 logs/cleanup.log

### Requirement 2: 手动依赖管理

**User Story:** 作为开发者，我希望明确管理项目依赖，避免版本冲突和不必要的库。

#### Acceptance Criteria

1. THE System SHALL 提供一个清理后的 requirements.txt 文件
2. THE requirements.txt SHALL 明确列出核心依赖及其版本范围
3. THE System SHALL 在文档中提供 flash-attn 的手动安装命令
4. THE System SHALL 提供依赖检查脚本，验证服务器上已安装的包
5. WHEN 依赖冲突时 THEN THE System SHALL 提示用户手动解决而非自动覆盖

### Requirement 3: 服务器路径适配

**User Story:** 作为开发者，我希望所有配置文件使用相对路径或服务器限制路径，避免路径违规。

#### Acceptance Criteria

1. THE System SHALL 检查所有 YAML 配置文件中的路径
2. WHEN 发现绝对路径 THEN THE System SHALL 转换为相对路径或 /home/Backup/maziheng 前缀
3. THE System SHALL 提供路径检查脚本 check_paths.py
4. THE System SHALL 在配置文件中添加路径说明注释
5. THE System SHALL 确保所有输出目录（checkpoints/, results/, logs/）使用相对路径

### Requirement 4: A100 优化配置

**User Story:** 作为开发者，我希望配置文件针对 A100 GPU 进行优化，充分利用硬件性能。

#### Acceptance Criteria

1. THE System SHALL 在 train_config.yaml 中设置合理的 batch_size 和 gradient_accumulation_steps
2. WHEN batch_size < 4 THEN THE System SHALL 确保 gradient_accumulation_steps >= 8
3. THE System SHALL 启用混合精度训练（bf16）
4. THE System SHALL 设置 num_workers 为 4-8 以充分利用 CPU
5. THE System SHALL 在配置文件注释中说明 Effective Batch Size 的计算方法

### Requirement 5: 简化测试策略

**User Story:** 作为开发者，我希望使用简单实用的测试方法，避免过度工程化。

#### Acceptance Criteria

1. THE System SHALL 提供 smoke_test.py 脚本用于快速验证核心功能
2. THE System SHALL 提供 test_preprocessing.py 用于数据预处理的单元测试
3. THE System SHALL 移除所有 hypothesis 属性测试相关代码
4. THE smoke_test.py SHALL 测试数据加载、模型初始化、前向传播
5. THE System SHALL 在文档中说明测试策略的简化原因

### Requirement 6: 噪声处理补充

**User Story:** 作为开发者，我希望补充噪声处理功能或在文档中说明为何跳过。

#### Acceptance Criteria

1. THE System SHALL 在 src/data/preprocessing.py 中添加可选的噪声处理函数
2. THE System SHALL 支持高斯滤波和中值滤波两种去噪方法
3. THE System SHALL 在配置文件中提供 enable_denoising 开关
4. THE System SHALL 在文档中说明医学影像数据集通常已预处理的原因
5. WHEN enable_denoising=False THEN THE System SHALL 跳过去噪步骤

### Requirement 7: 配置文件安全性检查

**User Story:** 作为开发者，我希望在训练前检查配置文件的合理性，避免常见错误。

#### Acceptance Criteria

1. THE System SHALL 提供 validate_config.py 脚本
2. THE System SHALL 检查 batch_size 和 gradient_accumulation_steps 的组合
3. THE System SHALL 检查所有路径是否在 /home/Backup/maziheng 范围内
4. THE System SHALL 检查 GPU 显存是否足够（基于配置估算）
5. WHEN 配置不合理 THEN THE System SHALL 提供具体的修改建议

### Requirement 8: 依赖版本兼容性检查

**User Story:** 作为开发者，我希望在安装依赖前检查服务器上已有的包，避免重复安装。

#### Acceptance Criteria

1. THE System SHALL 提供 check_dependencies.py 脚本
2. THE System SHALL 检查 PyTorch、CUDA 版本是否兼容
3. THE System SHALL 列出已安装的包及其版本
4. THE System SHALL 标记需要安装的缺失包
5. THE System SHALL 警告可能的版本冲突

### Requirement 9: 清理操作回滚机制

**User Story:** 作为开发者，我希望在清理操作失败时能够快速回滚，恢复原状。

#### Acceptance Criteria

1. THE System SHALL 在清理前创建操作日志
2. THE System SHALL 记录所有重命名和删除操作
3. WHEN 用户请求回滚 THEN THE System SHALL 根据日志恢复文件结构
4. THE System SHALL 提供 rollback.py 脚本
5. THE System SHALL 在回滚后验证项目结构完整性

### Requirement 10: 文档更新

**User Story:** 作为开发者，我希望文档反映简化后的实施方针，便于快速上手。

#### Acceptance Criteria

1. THE System SHALL 更新 README.md 说明简化后的项目结构
2. THE System SHALL 更新 SETUP.md 说明依赖安装步骤
3. THE System SHALL 创建 SERVER_SETUP.md 说明服务器环境配置
4. THE System SHALL 在文档中强调"避免过度工程化"的原则
5. THE System SHALL 提供快速启动指南（Quick Start）
