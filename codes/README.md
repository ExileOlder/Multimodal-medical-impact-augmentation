# Codes Directory

`codes/` 是当前仓库的有效代码入口，已经整理为 Stage A 1K final 方案。根目录旧版 `src/` 已删除。

## Main Entrypoints

| 文件 | 作用 |
| --- | --- |
| `gradio_demo.py` | 前端页面，默认选择 `../checkpoints/stageA_1k_final/adapter.pth` |
| `inference_mask.py` | 单张 `caption + fusion mask` 推理 |
| `train.py` | RetinaLogos 基座上的 Stage A 适配训练 |
| `generate_eval_samples.py` | 从 holdout metadata 批量生成评估样本 |
| `eval_structural_metrics.py` | 计算结构与颜色指标 |
| `compute_fid_kid.py` | 计算小样本 FID/KID 补充指标 |
| `run_final_example20_comparison.py` | 同批 examples_20 的 base-only/Ours 生成与评估 |
| `render_generation_comparisons.py` | 生成真实图、mask、生成图三联对照 |
| `configs/train/run_stagea_final_1k.sh` | 最终 1K 训练复现入口 |

## Final Adapter

默认权重在仓库根目录：

```text
../checkpoints/stageA_1k_final/adapter.pth
```

该 adapter 需要配合 RetinaLogos 基座权重：

```text
../checkpoints/consolidated.00-of-01.pth
```

基座权重、Gemma 文本编码器和 SDXL VAE 体积较大，不随仓库提交，需要在服务器本地放置。

## Frontend

```bash
python gradio_demo.py --host 0.0.0.0 --port 7860
```

页面已同步为 1K final 方案，默认加载最终 Stage A adapter。
前端和 CLI 推理默认 `mask_scale=1.0`。

## Training

```bash
bash configs/train/run_stagea_final_1k.sh
```

默认训练参数：

```text
MAX_STEPS=1000
COLOR_L1_WEIGHT=0.4
LUMA_LOSS_WEIGHT=0.2
COLOR_STAT_WEIGHT=0.1
```

继续训练到 1.5K-5K 的候选模型已经评估但未通过结构/颜色硬门槛，因此不作为默认入口。

## Inference

```bash
python inference_mask.py \
  --base_ckpt ../checkpoints \
  --adapter_ckpt ../checkpoints/stageA_1k_final/adapter.pth \
  --prompt "COLOR_STYLE: warm orange fundus; a color fundus photograph with clear retinal vessels." \
  --mask_path data/train/diabetic/mask/10000_left_fusion.png \
  --out_dir results/inference_stageA_1k_final \
  --image_size 512 \
  --num_sampling_steps 80 \
  --sampling_method euler \
  --cfg_scale 2.0 \
  --seed 42 \
  --precision bf16 \
  --qk_norm \
  --mask_scale 1.0 \
  --tokenizer_path google_gemma-2b \
  --local_diffusers_model_root sdxl-vae
```

## Final Examples-20 Comparison

```bash
python run_final_example20_comparison.py \
  --limit 20 \
  --num_sampling_steps 80 \
  --image_size 512 \
  --mode all \
  --out_root results/final_example20_comparison
```

该补充实验默认 `mask_scale=1.0`，并在同一批 20 个样本上计算结构/颜色指标与 FID/KID。
