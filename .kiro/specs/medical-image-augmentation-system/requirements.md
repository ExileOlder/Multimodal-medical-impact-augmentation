# Requirements Document

## Introduction

本文档定义了医学影像增广系统的功能需求。该系统基于现有的RetinaLogos项目（视网膜图像文本生成模型）进行扩展，旨在构建一个支持多模态输入（文本、图像、分割掩码）的医学影像生成与增广平台。系统将通过融合语义文本和结构信息，生成高质量、医学逻辑一致的视网膜图像，用于数据增广和下游任务性能提升。

## Glossary

- **System**: 医学影像增广系统（Medical Image Augmentation System）
- **NexDiT_Model**: 基于Lumina-Next架构的Diffusion Transformer模型
- **Segmentation_Mask**: 分割掩码，用于标识图像中不同解剖结构或病理区域的像素级标注
- **Multimodal_Input**: 多模态输入，包含文本描述（caption）、参考图像和分割掩码
- **Data_Preprocessor**: 数据预处理器，负责图像标准化、噪声处理和数据格式转换
- **Structure_Encoder**: 结构编码器，将分割掩码编码为模型可用的结构先验信息
- **Web_Interface**: Web用户界面，提供多模态输入管理和生成结果展示功能
- **Quality_Evaluator**: 质量评估器，计算生成图像的质量指标和结构保真度
- **Training_Pipeline**: 训练流程，包含模型训练、验证和超参数配置
- **Generation_Engine**: 生成引擎，执行图像生成任务并应用质量控制策略

## Requirements

### Requirement 1: 多模态数据格式支持

**User Story:** 作为数据科学家，我希望系统能够处理包含图像、文本描述和分割掩码的多模态数据，以便构建结构化的训练数据集。

#### Acceptance Criteria

1. WHEN 用户提供JSON或JSONL格式的数据文件 THEN THE Data_Preprocessor SHALL 解析文件并提取image、caption和mask字段
2. WHEN 数据文件中缺少mask字段 THEN THE Data_Preprocessor SHALL 将该条数据标记为仅文本模式并继续处理
3. WHEN 分割掩码的尺寸与对应图像不匹配 THEN THE Data_Preprocessor SHALL 返回明确的错误信息并指出不匹配的数据项
4. THE Data_Preprocessor SHALL 支持批量导入多个数据文件
5. WHEN 数据加载完成 THEN THE System SHALL 生成数据统计报告，包含总样本数、包含掩码的样本数和数据格式分布

### Requirement 2: 图像预处理与标准化

**User Story:** 作为研究人员，我希望系统能够自动标准化医学影像数据，以便确保训练数据的一致性和质量。

#### Acceptance Criteria

1. WHEN 输入图像的分辨率不一致 THEN THE Data_Preprocessor SHALL 将所有图像调整到统一的目标分辨率
2. THE Data_Preprocessor SHALL 对输入图像执行像素值归一化处理
3. WHEN 图像包含明显噪声 THEN THE Data_Preprocessor SHALL 应用降噪算法并保留医学细节特征
4. WHEN 分割掩码包含多个类别 THEN THE Data_Preprocessor SHALL 验证类别标签的有效性并生成类别映射表
5. THE Data_Preprocessor SHALL 保存预处理后的数据及其元数据到指定输出目录

### Requirement 3: 结构信息编码与融合

**User Story:** 作为模型开发者，我希望将分割掩码作为结构先验融入生成模型，以便生成的图像符合解剖结构约束。

#### Acceptance Criteria

1. THE Structure_Encoder SHALL 将分割掩码编码为与图像特征维度兼容的张量表示
2. WHEN NexDiT_Model 接收文本和掩码输入 THEN THE System SHALL 在模型的注意力机制中融合文本特征和结构特征
3. THE System SHALL 实现ControlNet风格的条件控制机制或等效的结构融合方法
4. WHEN 仅提供文本输入而无掩码 THEN THE NexDiT_Model SHALL 降级为纯文本到图像生成模式
5. THE Structure_Encoder SHALL 支持多尺度掩码特征提取以匹配扩散模型的多分辨率处理

### Requirement 4: 模型架构配置与扩展

**User Story:** 作为机器学习工程师，我希望能够灵活配置模型架构参数，以便针对不同的医学影像任务进行优化。

#### Acceptance Criteria

1. THE System SHALL 提供配置文件接口用于设置NexDiT_Model的层数、隐藏维度和注意力头数
2. WHEN 用户修改模型配置 THEN THE System SHALL 验证参数的有效性并在无效时返回具体错误信息
3. THE System SHALL 支持加载预训练的RetinaLogos权重作为初始化
4. WHEN 启用结构融合模式 THEN THE System SHALL 自动添加必要的结构编码器模块到模型架构中
5. THE System SHALL 记录所有架构配置到模型检查点文件中以确保可重现性

### Requirement 5: 训练流程与超参数管理

**User Story:** 作为研究人员，我希望系统提供完整的训练流程管理，以便高效地训练和优化生成模型。

#### Acceptance Criteria

1. THE Training_Pipeline SHALL 支持通过配置文件设置学习率、批次大小、训练轮数和优化器类型
2. WHEN 训练过程中 THEN THE System SHALL 每N个迭代保存模型检查点并记录训练损失
3. THE Training_Pipeline SHALL 实现验证集评估并计算生成图像的质量指标
4. WHEN GPU内存不足 THEN THE System SHALL 返回明确的错误信息并建议减小批次大小
5. THE Training_Pipeline SHALL 支持从中断的检查点恢复训练

### Requirement 6: 一致性约束与质量控制

**User Story:** 作为医学影像专家，我希望生成的图像在结构、语义和视觉质量上保持一致性，以便确保医学逻辑的正确性。

#### Acceptance Criteria

1. WHEN 生成图像时 THEN THE Generation_Engine SHALL 应用结构一致性损失以确保生成结果与输入掩码对齐
2. THE Quality_Evaluator SHALL 计算生成图像与参考图像之间的结构相似度指标
3. WHEN 生成结果的结构偏差超过阈值 THEN THE System SHALL 标记该样本为低质量并记录到日志
4. THE Generation_Engine SHALL 支持多种采样策略（DDPM、DDIM等）以平衡质量和生成速度
5. THE System SHALL 实现语义一致性验证，确保生成图像的病理特征与文本描述匹配

### Requirement 7: Web界面与多模态输入管理

**User Story:** 作为临床研究人员，我希望通过直观的Web界面上传数据并查看生成结果，以便无需编程即可使用系统。

#### Acceptance Criteria

1. THE Web_Interface SHALL 提供文件上传功能，支持图像（PNG、JPEG）、文本和掩码（PNG、NPY）格式
2. WHEN 用户上传文件 THEN THE System SHALL 验证文件格式和大小，并在无效时显示错误提示
3. THE Web_Interface SHALL 显示上传的图像、掩码叠加可视化和文本描述的预览
4. WHEN 用户提交生成请求 THEN THE System SHALL 显示生成进度并在完成后展示结果图像
5. THE Web_Interface SHALL 允许用户下载生成的图像和对应的质量评估报告

### Requirement 8: 生成结果展示与对比分析

**User Story:** 作为研究人员，我希望系统能够展示生成结果并提供详细的质量分析，以便评估增广效果。

#### Acceptance Criteria

1. THE Web_Interface SHALL 并排显示输入图像、输入掩码、生成图像和结构叠加对比
2. WHEN 生成完成 THEN THE System SHALL 计算并显示PSNR、SSIM、FID等图像质量指标
3. THE Web_Interface SHALL 提供交互式的掩码-图像叠加可视化，支持透明度调节
4. THE System SHALL 生成结构保真度热图，高亮显示结构偏差区域
5. THE Web_Interface SHALL 支持批量生成模式并展示所有结果的统计摘要

### Requirement 9: 结构一致性验证

**User Story:** 作为质量控制人员，我希望系统能够自动验证生成图像的结构一致性，以便筛选出符合医学标准的增广数据。

#### Acceptance Criteria

1. THE Quality_Evaluator SHALL 对生成图像执行自动分割并与输入掩码进行对比
2. WHEN 结构一致性评分低于设定阈值 THEN THE System SHALL 将该样本标记为不合格
3. THE Quality_Evaluator SHALL 计算每个解剖区域的Dice系数和IoU指标
4. THE System SHALL 生成结构一致性报告，包含整体评分和区域级别的详细分析
5. THE Web_Interface SHALL 可视化显示结构差异区域并提供改进建议

### Requirement 10: 下游任务性能评估

**User Story:** 作为数据科学家，我希望评估增广数据对下游任务的影响，以便验证系统的实用价值。

#### Acceptance Criteria

1. THE System SHALL 提供接口导出增广数据集用于外部分类或分割任务
2. WHEN 用户提供下游任务的性能指标 THEN THE System SHALL 记录并对比使用增广前后的性能变化
3. THE System SHALL 支持配置增广数据与真实数据的混合比例
4. THE System SHALL 生成增广效果评估报告，包含任务性能提升的统计分析
5. THE Web_Interface SHALL 展示增广数据对不同下游任务的贡献度分析

### Requirement 11: 配置管理与实验追踪

**User Story:** 作为研究人员，我希望系统能够管理不同的实验配置并追踪结果，以便进行系统化的实验对比。

#### Acceptance Criteria

1. THE System SHALL 支持通过YAML或JSON文件定义完整的实验配置
2. WHEN 启动训练或生成任务 THEN THE System SHALL 自动记录所有配置参数到实验日志
3. THE System SHALL 为每个实验分配唯一标识符并组织输出文件到对应目录
4. THE System SHALL 提供命令行接口用于列出、对比和加载历史实验配置
5. THE Web_Interface SHALL 展示实验历史记录并支持配置的复制和修改

### Requirement 12: 错误处理与日志记录

**User Story:** 作为系统管理员，我希望系统能够提供详细的错误信息和日志，以便快速定位和解决问题。

#### Acceptance Criteria

1. WHEN 系统遇到错误 THEN THE System SHALL 返回包含错误类型、位置和建议解决方案的消息
2. THE System SHALL 将所有操作日志记录到文件，包含时间戳、操作类型和执行状态
3. WHEN 模型推理失败 THEN THE System SHALL 记录输入数据的元信息以便复现问题
4. THE System SHALL 区分不同级别的日志（DEBUG、INFO、WARNING、ERROR）
5. THE Web_Interface SHALL 提供日志查看功能并支持按时间和级别过滤
