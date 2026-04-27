# Multimodal Medical Image Augmentation

本仓库是“基于多模态信息融合的医学影像增广系统”的最终整理版。当前默认方案是 RetinaLogos 基座模型上的 **Stage A 1K fusion-mask adapter**：以医学 caption 和六通道结构融合 mask 为条件生成眼底彩照，并用结构指标、颜色指标和小样本 FID/KID 做补充评估。

继续从 1K 权重训练到 1.5K-5K 的候选模型已经完成评估，但没有同时满足“结构不能退化、颜色不能偏紫或偏离原图”的守门条件，因此最终冻结 1K adapter。

## Current Mainline

| 项目 | 当前设置 |
| --- | --- |
| 任务 | `caption + fusion mask -> fundus image` |
| 基座 | RetinaLogos / Lumina-Next style text-to-image backbone |
| 默认 adapter | `checkpoints/stageA_1k_final/adapter.pth` |
| 默认前端 | `codes/gradio_demo.py` |
| 默认推理脚本 | `codes/inference_mask.py` |
| 结构 mask 通道 | `6` |
| 默认 `mask_scale` | `1.0` |

Stage B lesion-crop editing 仍定位为探索性扩展，不是最终前端和正式量化主线。

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── checkpoints/stageA_1k_final/
│   ├── adapter.pth
│   ├── args.json
│   ├── run_manifest.json
│   ├── structural_metrics.json
│   └── triptych_sheet_canonical.png
└── codes/
    ├── gradio_demo.py
    ├── inference_mask.py
    ├── train.py
    ├── generate_eval_samples.py
    ├── eval_structural_metrics.py
    ├── compute_fid_kid.py
    ├── run_structural_eval.py
    ├── run_final_example20_comparison.py
    ├── render_generation_comparisons.py
    ├── struct_mask_utils.py
    ├── configs/
    ├── data/
    ├── models/
    ├── transport/
    └── utils/
```

`codes/` 是唯一有效代码入口。旧版 `src/` 和本地工作台式目录已清理。

## Local Artifacts

仓库提交的是代码、小型配置、最终 Stage A adapter 和少量说明文件。以下内容需要在训练/推理服务器本地准备，不应提交到 Git：

```text
checkpoints/consolidated.00-of-01.pth
codes/google_gemma-2b/
codes/sdxl-vae/
codes/data/train/
codes/data/test/
codes/data/merged/
codes/data/derived/
codes/results/
```

## Environment

```bash
cd codes
conda env create -f environment_RetinaLogos.yml
conda activate RetinaLogos
```

也可以按需安装精简依赖：

```bash
pip install -r requirements.txt
```

`flash-attn` 需要匹配 CUDA/PyTorch 环境，建议单独确认后安装。

## Frontend

```bash
cd codes
python gradio_demo.py --host 0.0.0.0 --port 7860
```

前端默认使用：

- base checkpoint: `../checkpoints`
- selected adapter: `../checkpoints/stageA_1k_final/adapter.pth`
- tokenizer: `google_gemma-2b`
- VAE: `sdxl-vae`
- image size: `512`
- sampling steps: `80`
- CFG: `2.0`
- mask scale: `1.0`

上传 fusion mask 时走 Stage A 多模态路径；不上传 mask 时可退回 base-only 文本生成。

## CLI Inference

```bash
cd codes
python inference_mask.py \
  --model NextDiT_2B_GQA_patch2 \
  --base_ckpt ../checkpoints \
  --adapter_ckpt ../checkpoints/stageA_1k_final/adapter.pth \
  --prompt "COLOR_STYLE: warm orange fundus; a color fundus photograph with clear optic disc and retinal vessels." \
  --mask_path data/train/diabetic/mask/10000_left_fusion.png \
  --out_dir results/inference_stageA_1k_final \
  --image_size 512 \
  --num_sampling_steps 80 \
  --sampling_method euler \
  --cfg_scale 2.0 \
  --mask_scale 1.0 \
  --struct_mask_channels 6 \
  --seed 42 \
  --precision bf16 \
  --qk_norm \
  --tokenizer_path google_gemma-2b \
  --local_diffusers_model_root sdxl-vae
```

## Training

最终 1K adapter 的训练入口：

```bash
cd codes
bash configs/train/run_stagea_final_1k.sh
```

默认训练配置：

```text
RUN_NAME=mainline_stageA_global_1k_maskqc_clean_canonical_strictalign_colorstyle8_compact_frombase_colorfix
MAX_STEPS=1000
COLOR_L1_WEIGHT=0.4
LUMA_LOSS_WEIGHT=0.2
COLOR_STAT_WEIGHT=0.1
```

## Evaluation

重新生成 holdout 评估样本：

```bash
cd codes
python generate_eval_samples.py \
  --metadata data/derived/diabetic_merged/metadata_example_20_maskqc_clean_colorstyle8.jsonl \
  --root_dir . \
  --base_ckpt ../checkpoints \
  --adapter_ckpt ../checkpoints/stageA_1k_final/adapter.pth \
  --out_dir results/eval_stageA_1k_final/generated \
  --limit 20 \
  --cfg_scale 2.0 \
  --mask_scale 1.0 \
  --num_sampling_steps 80 \
  --precision bf16
```

同一批 `examples_20` 的 base-only/Ours 补充实验：

```bash
cd codes
python run_final_example20_comparison.py \
  --limit 20 \
  --num_sampling_steps 80 \
  --image_size 512 \
  --mode all \
  --out_root results/final_example20_comparison
```

这次补充实验使用 `mask_scale=1.0`，不是 2.0。

## Supplementary Results

在同一批 20 个 `examples_20` 样本上，Stage A fusion-mask 方法相对 RetinaLogos base-only 的补充结果如下：

| Metric | Base-only | Stage A | Better |
| --- | ---: | ---: | --- |
| Global SSIM | 0.616187 | 0.784549 | higher |
| Global PSNR | 13.835871 | 16.589129 | higher |
| Mask SSIM | 0.304930 | 0.678961 | higher |
| Mask PSNR | 12.669021 | 15.267288 | higher |
| Fundus mean abs error | 0.153380 | 0.125971 | lower |
| Fundus chroma L1 | 0.046107 | 0.041244 | lower |
| Fundus R-B gap abs error | 0.127556 | 0.100155 | lower |
| FID | 157.189926 | 127.594292 | lower |
| KID mean | 0.130806 | 0.092133 | lower |

该实验只有 20 个样本，FID/KID 应表述为小样本补充验证，不应夸大为大规模分布评估。

## Why Stage A Can Still Underperform A Baseline

Stage A 虽然从 RetinaLogos 基座继续训练，但 adapter 会把生成分布拉向较小的 mask/caption 训练子集。若训练样本覆盖面、mask 质量、颜色分布、损失权重或训练步数不平衡，模型可能牺牲基座原本的自然图像分布质量来服从结构条件。因此“从基线训练而来”不等于“所有图像质量指标必然高于基线”。

颜色一致性也不是有 caption 就能自动解决。颜色是密集、低层的连续图像统计，caption 只是离散语义 token；即使加入 `COLOR_STYLE`，模型仍可能受数据颜色偏置、VAE 解码、预处理亮度、采样随机性和 adapter 容量影响。最终版本通过 compact color-style caption、颜色损失和 checkpoint guard 缓解该问题，但 caption 本身不能完全替代颜色分布约束。
