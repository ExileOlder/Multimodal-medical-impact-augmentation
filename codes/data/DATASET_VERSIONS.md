# Diabetic 数据版本说明

## official 版本

- 路径：`codes/data/train/diabetic/autodl`
- 原图来源：仅 `codes/data/train/diabetic/source/train`
- 作用：保留一套更规范的 train/val 切分，适合论文正文和过程汇报

当前统计：

- `caption_records`: 27576
- `mask_files`: 18809
- `matched_records`: 7477
- `train_records`: 7104
- `val_records`: 373

说明：

- 这套版本只使用 train 原图目录
- 因为 caption 文件中还有一部分样本的原图落在 test 目录，所以 official 版本不会覆盖全部可用三元样本

## merged 版本

- 路径：`codes/data/merged/diabetic/autodl`
- 原图来源：
  - `codes/data/train/diabetic/source/train`
  - `codes/data/test/diabetic/source/test`
- 作用：作为生成模型训练的扩展版本，尽可能利用全部已有 `caption + mask + image` 三元样本

当前统计：

- `caption_records`: 27576
- `mask_files`: 18809
- `matched_records`: 18809
- `train_records`: 17869
- `val_records`: 940

说明：

- merged 版本使用 train 与 test 两个原图目录联合构建
- 这样可以把 caption 文件中落在 test 原图目录里的 11332 条可配对样本也纳入训练候选
- 为避免和 official 版本混淆，merged 版本单独输出到 `codes/data/merged/diabetic/autodl`

## 当前建议

- 论文正文和阶段汇报优先引用 official 版本
- 生成模型正式训练优先使用 merged 版本
- 后续所有实验记录必须明确写明使用的是 `official` 还是 `merged`
