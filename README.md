# Multimodal Medical Image Augmentation

本仓库是“基于多模态信息融合的医学影像增广系统”的最终整理版，当前默认方案固定为 **Stage A 1K colorfix adapter**。继续从 1K 权重训练到 1.5K-5K 的候选模型已经完成评估，但没有同时满足“结构不能退化、颜色不能偏紫或偏离原图”的硬性要求，因此不作为最终权重。

## Final Checkpoint

| 项目 | 选择 |
| --- | --- |
| 默认权重 | `checkpoints/stageA_1k_final/adapter.pth` |
| 训练步数 | `0001000` |
| 基座模型 | RetinaLogos `consolidated.00-of-01.pth` |
| 默认前端 | `codes/gradio_demo.py` |

1K 评估指标：

| metric | value |
| --- | ---: |
| `global_ssim.mean` | `0.7728` |
| `mask_ssim.mean` | `0.6632` |
| `global_l1.mean` | `0.1314` |
| `mask_l1.mean` | `0.1678` |
| `fundus_mean_abs_error.mean` | `0.1462` |
| `fundus_chroma_l1.mean` | `0.0571` |

继续训练候选 `0001500`、`0002000`、`0002500`、`0003000`、`0003500`、`0004000`、`0004500`、`0005000` 均未通过守门条件，所以最终冻结 1K。

## Repository Layout

```text
.
├── codes/                         # RetinaLogos 适配、训练、推理、评估和 Gradio 前端
│   ├── gradio_demo.py             # 已同步为 1K final adapter 的前端
│   ├── inference_mask.py          # 单样本 mask/caption 推理
│   ├── train.py                   # Stage A 训练入口
│   ├── generate_eval_samples.py   # holdout 样本生成
│   ├── eval_structural_metrics.py # 结构/颜色指标
│   ├── run_structural_eval.py     # 自动评估编排
│   ├── configs/train/
│   │   ├── run_stagea_final_1k.sh
│   │   └── run_stagea_colorstyle8_compact_frombase.sh
│   └── models/, data/, transport/, utils/
├── checkpoints/stageA_1k_final/
│   ├── adapter.pth
│   ├── structural_metrics.json
│   ├── triptych_sheet_canonical.png
│   ├── args.json
│   └── run_manifest.json
└── requirements.txt
```

根目录 `src/` 已移除，当前仓库以 `codes/` 作为唯一有效代码入口。

## Assets Not Stored Here

仓库中保存的是 1K adapter 和评估产物。以下大文件需要在服务器本地准备：

```text
checkpoints/consolidated.00-of-01.pth      # RetinaLogos 基座权重
codes/google_gemma-2b/                     # Gemma tokenizer/text encoder
codes/sdxl-vae/                            # 本地 SDXL VAE，可选但推荐离线放置
codes/data/...                             # 训练/评估数据与 fusion mask
```

本地训练服务器上的推荐布局与前端默认路径一致。

## Environment

推荐直接使用导出的 conda 环境：

```bash
cd codes
conda env create -f environment_RetinaLogos.yml
conda activate RetinaLogos
```

也可以按需安装精简依赖：

```bash
pip install -r requirements.txt
```

`flash-attn` 需要 CUDA 编译环境，建议在 PyTorch/CUDA 确认无误后单独安装。

## Gradio Frontend

```bash
cd codes
python gradio_demo.py --host 0.0.0.0 --port 7860
```

页面默认指向：

```text
Base checkpoint: ../checkpoints/consolidated.00-of-01.pth
Selected adapter: ../checkpoints/stageA_1k_final/adapter.pth
Preview: ../checkpoints/stageA_1k_final/triptych_sheet_canonical.png
```

可上传 fusion mask 和可选颜色参考图。若不提供 mask，前端会以禁用结构条件的方式调用推理脚本；正式评估建议提供 fusion mask。

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
  --seed 42 \
  --precision bf16 \
  --qk_norm \
  --mask_scale 1.0 \
  --tokenizer_path google_gemma-2b \
  --local_diffusers_model_root sdxl-vae
```

## Reproduce Training

最终方案的训练入口：

```bash
cd codes
bash configs/train/run_stagea_final_1k.sh
```

该脚本默认：

```text
RUN_NAME=mainline_stageA_global_1k_maskqc_clean_canonical_strictalign_colorstyle8_compact_frombase_colorfix
MAX_STEPS=1000
COLOR_L1_WEIGHT=0.4
LUMA_LOSS_WEIGHT=0.2
COLOR_STAT_WEIGHT=0.1
```

## Evaluation

1K 的评估结果已经整理到 `checkpoints/stageA_1k_final/`。如需重新生成 holdout 评估，可使用：

```bash
cd codes
python generate_eval_samples.py \
  --metadata data/derived/diabetic_merged/metadata_example_20_maskqc_clean_colorstyle8.jsonl \
  --root_dir . \
  --base_ckpt ../checkpoints \
  --adapter_ckpt ../checkpoints/stageA_1k_final/adapter.pth \
  --out_dir results/eval_stageA_1k_final/generated \
  --limit 10 \
  --cfg_scale 2.0 \
  --num_sampling_steps 80 \
  --precision bf16
```

随后使用 `eval_structural_metrics.py` 与 `render_generation_comparisons.py` 计算指标并渲染三联图。
