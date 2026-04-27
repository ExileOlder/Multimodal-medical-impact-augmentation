import argparse
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

from struct_mask_utils import augment_caption_with_struct_hints


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent

DEFAULT_BASE_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_FINAL_ADAPTER = REPO_ROOT / "checkpoints" / "stageA_1k_final" / "adapter.pth"
DEFAULT_TOKENIZER_DIR = PROJECT_DIR / "google_gemma-2b"
DEFAULT_VAE_PATH = PROJECT_DIR / "sdxl-vae"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "inference_stageA_1k_final"

FINAL_RUN_NAME = "mainline_stageA_global_1k_maskqc_clean_canonical_strictalign_colorstyle8_compact_frombase_colorfix"
DEFAULT_LANGUAGE = "en"
DEFAULT_STRUCT_MASK_CHANNELS = 6

LANG_TEXTS = {
    "en": {
        "title": "RetinaLogos Multimodal Fundus Generator",
        "subtitle": (
            "Text-only mode uses the RetinaLogos base checkpoint. Multimodal mode adds the selected "
            "Stage A adapter and an optional structure mask."
        ),
        "badge_text_only": "Text-only base mode",
        "badge_mask_optional": "Mask condition optional",
        "badge_backend": "Backend: inference_mask.py",
        "language": "Language",
        "language_en": "English",
        "language_zh": "中文",
        "model": "Model",
        "conditions": "Conditions",
        "settings": "Sampling",
        "base_ckpt": "Base checkpoint directory",
        "adapter_ckpt": "Selected 1K adapter (mask mode only)",
        "tokenizer": "Tokenizer directory",
        "vae": "Local VAE path",
        "output": "Output directory",
        "prompt": "Caption",
        "prompt_placeholder": "Enter a fundus caption.",
        "mask": "Structure mask",
        "use_color_reference": "Use color reference",
        "color_reference": "Color reference (optional)",
        "image_size": "Image size",
        "steps": "Sampling steps",
        "cfg": "CFG scale",
        "mask_scale": "Mask scale",
        "seed": "Seed",
        "samples": "Samples",
        "precision": "Precision",
        "gpu": "CUDA_VISIBLE_DEVICES",
        "run": "Generate",
        "gallery": "Generated images",
        "summary": "Run summary",
        "logs": "Logs",
        "error_caption_empty": "Caption cannot be empty.",
        "error_gpu_empty": "CUDA_VISIBLE_DEVICES cannot be empty.",
        "error_inference_failed": "Inference failed for seed {seed}.",
        "error_no_image": "Inference finished but no image was produced: {path}",
        "summary_final_run": "Final run",
        "summary_base_ckpt": "Base checkpoint",
        "summary_adapter_ckpt": "Adapter checkpoint",
        "summary_mode": "Mode",
        "summary_mode_base": "text-only base",
        "summary_mode_mm": "multimodal mask+caption",
        "summary_tokenizer": "Tokenizer",
        "summary_vae": "VAE",
        "summary_mask": "Mask",
        "summary_color_reference": "Color reference",
        "summary_prompt_processing": "Prompt processing",
        "summary_struct_hints_enabled": "structure hints enabled",
        "summary_struct_hints_disabled": "raw caption",
        "summary_image_size": "Image size",
        "summary_steps": "Sampling steps",
        "summary_cfg": "CFG scale",
        "summary_mask_scale": "Mask scale",
        "summary_seed_range": "Seed range",
        "summary_precision": "Precision",
        "summary_gpu": "GPU IDs",
        "no_mask": "disabled",
        "no_color_reference": "disabled",
    },
    "zh": {
        "title": "RetinaLogos 多模态眼底生成器",
        "subtitle": "文本模式使用 RetinaLogos 基座权重；多模态模式支持可选结构掩膜。",
        "badge_text_only": "纯文本基座模式",
        "badge_mask_optional": "掩膜条件可选",
        "badge_backend": "后端：inference_mask.py",
        "language": "语言",
        "language_en": "English",
        "language_zh": "中文",
        "model": "模型",
        "conditions": "条件",
        "settings": "采样",
        "base_ckpt": "基座权重目录",
        "adapter_ckpt": "1K 适配器（仅掩膜模式使用）",
        "tokenizer": "分词器目录",
        "vae": "本地 VAE 路径",
        "output": "输出目录",
        "prompt": "描述",
        "prompt_placeholder": "请输入眼底图像描述。",
        "mask": "结构掩膜",
        "use_color_reference": "使用颜色参考",
        "color_reference": "颜色参考（可选）",
        "image_size": "图像尺寸",
        "steps": "采样步数",
        "cfg": "CFG 强度",
        "mask_scale": "掩膜强度",
        "seed": "随机种子",
        "samples": "生成张数",
        "precision": "精度",
        "gpu": "CUDA_VISIBLE_DEVICES",
        "run": "生成",
        "gallery": "生成结果",
        "summary": "运行摘要",
        "logs": "日志",
        "error_caption_empty": "描述不能为空。",
        "error_gpu_empty": "CUDA_VISIBLE_DEVICES 不能为空。",
        "error_inference_failed": "种子 {seed} 的推理失败。",
        "error_no_image": "推理结束但未生成图像：{path}",
        "summary_final_run": "最终运行",
        "summary_base_ckpt": "基座权重",
        "summary_adapter_ckpt": "适配器权重",
        "summary_mode": "模式",
        "summary_mode_base": "纯文本基座",
        "summary_mode_mm": "多模态 掩膜+描述",
        "summary_tokenizer": "分词器",
        "summary_vae": "VAE",
        "summary_mask": "掩膜",
        "summary_color_reference": "颜色参考",
        "summary_prompt_processing": "描述处理",
        "summary_struct_hints_enabled": "已启用结构提示增强",
        "summary_struct_hints_disabled": "原始描述",
        "summary_image_size": "图像尺寸",
        "summary_steps": "采样步数",
        "summary_cfg": "CFG 强度",
        "summary_mask_scale": "掩膜强度",
        "summary_seed_range": "种子范围",
        "summary_precision": "精度",
        "summary_gpu": "GPU 编号",
        "no_mask": "已禁用",
        "no_color_reference": "已禁用",
    },
}

APP_CSS = """
.gradio-container {
  background: #f7f8f6;
  color: #1f2933;
}
.app-shell {
  max-width: 1320px;
  margin: 0 auto;
}
.hero-panel,
.control-panel,
.result-panel {
  border-radius: 8px;
  border: 1px solid #d8ded8;
  background: #ffffff;
}
.hero-panel {
  padding: 18px 20px;
  position: relative;
}
.hero-layout {
  display: block;
}
.hero-copy {
  text-align: center;
  padding: 0 120px;
}
.hero-title {
  font-size: 28px;
  line-height: 1.25;
  font-weight: 800;
  color: #173f35;
}
.hero-subtitle {
  margin-top: 8px;
  line-height: 1.6;
  color: #44515c;
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
  justify-content: center;
}
.hero-lang {
  position: absolute;
  right: 18px;
  bottom: 14px;
  width: 96px;
}
.hero-lang label {
  display: none !important;
}
.hero-lang .form,
.hero-lang .block {
  min-width: 0 !important;
  padding: 0 !important;
}
.hero-lang [data-testid="block-info"] {
  display: none !important;
}
.hero-lang input {
  font-size: 12px !important;
}
.badge {
  display: inline-flex;
  align-items: center;
  padding: 6px 9px;
  border-radius: 8px;
  background: #eef5f1;
  border: 1px solid #d6e5dc;
  color: #214d42;
  font-size: 13px;
}
.badge.warm {
  background: #fff3e5;
  border-color: #f1d7b8;
  color: #70461d;
}
.badge.rose {
  background: #f8eeee;
  border-color: #ead0d0;
  color: #6a3434;
}
.control-panel,
.result-panel {
  padding: 14px;
}
.run-button button {
  height: 48px;
  border-radius: 8px !important;
  border: 0 !important;
  background: #1f6f61 !important;
  color: #fff !important;
  font-weight: 700 !important;
}
@media (max-width: 900px) {
  .hero-layout {
    display: flex;
    flex-direction: column;
  }
  .hero-copy {
    padding: 0;
  }
  .hero-lang {
    position: static;
    width: 96px;
    align-self: center;
    margin-top: 10px;
  }
}
"""


def existing_value(path: Path) -> str:
    return str(path.resolve()) if path.exists() else str(path)


def get_texts(language: str) -> dict[str, str]:
    return LANG_TEXTS.get(language, LANG_TEXTS[DEFAULT_LANGUAGE])


def hero_html(language: str) -> str:
    texts = get_texts(language)
    return (
        '<div class="hero-copy">'
        f'<div class="hero-title">{texts["title"]}</div>'
        f'<div class="hero-subtitle">{texts["subtitle"]}</div>'
        '<div class="badge-row">'
        f'<span class="badge">{texts["badge_text_only"]}</span>'
        f'<span class="badge rose">{texts["badge_mask_optional"]}</span>'
        f'<span class="badge">{texts["badge_backend"]}</span>'
        "</div></div>"
    )


def language_key(choice: str) -> str:
    return "zh" if choice == LANG_TEXTS["zh"]["language_zh"] else DEFAULT_LANGUAGE


def strip_color_style_prompt(prompt: str) -> str:
    stripped = prompt.strip()
    if not stripped.upper().startswith("COLOR_STYLE:"):
        return prompt

    section_names = (
        "GLOBAL_SUMMARY",
        "AGE_APPEARANCE",
        "PIGMENTATION",
        "MICROANEURYSMS",
        "HEMORRHAGES",
        "EXUDATES",
        "SOFT_EXUDATES",
        "MACULA_FOVEA",
        "OPTIC_DISC",
        "VESSELS",
        "IMPRESSION",
    )
    for name in section_names:
        marker = f" {name}:"
        idx = stripped.find(marker)
        if idx > 0:
            return stripped[idx + 1 :].lstrip()

    if ". " in stripped:
        return stripped.split(". ", 1)[1].lstrip()
    return prompt


def language_updates(choice: str):
    texts = get_texts(language_key(choice))
    return [
        gr.update(value=hero_html(language_key(choice))),
        gr.update(label=texts["prompt"], placeholder=texts["prompt_placeholder"]),
        gr.update(label=texts["model"]),
        gr.update(label=texts["base_ckpt"]),
        gr.update(label=texts["adapter_ckpt"]),
        gr.update(label=texts["tokenizer"]),
        gr.update(label=texts["vae"]),
        gr.update(label=texts["output"]),
        gr.update(label=texts["conditions"]),
        gr.update(label=texts["mask"]),
        gr.update(label=texts["use_color_reference"]),
        gr.update(label=texts["color_reference"]),
        gr.update(label=texts["settings"]),
        gr.update(label=texts["image_size"]),
        gr.update(label=texts["precision"]),
        gr.update(label=texts["steps"]),
        gr.update(label=texts["cfg"]),
        gr.update(label=texts["mask_scale"]),
        gr.update(label=texts["seed"]),
        gr.update(label=texts["samples"]),
        gr.update(label=texts["gpu"]),
        gr.update(value=texts["run"]),
        gr.update(label=texts["gallery"]),
        gr.update(label=texts["summary"]),
        gr.update(label=texts["logs"]),
    ]


def resolve_ckpt_file(path_text: str, default_name: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if path.is_dir():
        path = path / default_name
    if not path.exists():
        raise gr.Error(f"{label} not found: {path}")
    return path


def resolve_required_dir(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise gr.Error(f"{label} not found: {path}")
    return path


def optional_existing_path(path_text: str, label: str) -> str:
    value = (path_text or "").strip()
    if not value:
        return ""
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise gr.Error(f"{label} not found: {path}")
    return str(path)


def run_single_sample(
    prompt: str,
    base_ckpt: Path,
    adapter_ckpt: Path | None,
    tokenizer_dir: Path,
    vae_path: str,
    output_dir: Path,
    mask_path: str | None,
    color_reference: str | None,
    texts: dict[str, str],
    image_size: int,
    sampling_steps: int,
    cfg_scale: float,
    mask_scale: float,
    seed: int,
    precision: str,
    index: int,
    gpu_ids: str,
) -> tuple[str, str]:
    output_name = f"stageA1k_final_seed{seed}_{index:02d}.png"
    command = [
        sys.executable,
        "inference_mask.py",
        "--model",
        "NextDiT_2B_GQA_patch2",
        "--base_ckpt",
        str(base_ckpt),
        "--prompt",
        prompt,
        "--out_dir",
        str(output_dir),
        "--image_size",
        str(image_size),
        "--num_sampling_steps",
        str(sampling_steps),
        "--sampling_method",
        "euler",
        "--cfg_scale",
        str(cfg_scale),
        "--seed",
        str(seed),
        "--precision",
        precision,
        "--qk_norm",
        "--mask_scale",
        str(mask_scale),
        "--tokenizer_path",
        str(tokenizer_dir),
        "--struct_mask_channels",
        str(DEFAULT_STRUCT_MASK_CHANNELS),
        "--output_name",
        output_name,
    ]
    if adapter_ckpt is not None:
        command.extend(["--adapter_ckpt", str(adapter_ckpt)])
    if vae_path:
        command.extend(["--local_diffusers_model_root", vae_path])
    if mask_path:
        command.extend(["--mask_path", mask_path])
    else:
        command.append("--disable_mask_condition")
    if color_reference:
        command.extend(["--fundus_color_reference", color_reference])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    process = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    logs = (process.stdout or "") + ("\n" + process.stderr if process.stderr else "")
    out_path = output_dir / output_name
    if process.returncode != 0:
        raise gr.Error(f"{texts['error_inference_failed'].format(seed=seed)}\n\n{logs}")
    if not out_path.exists():
        raise gr.Error(f"{texts['error_no_image'].format(path=out_path)}\n\n{logs}")
    return str(out_path), logs


def run_generation(
    language_choice: str,
    prompt: str,
    base_checkpoint_dir: str,
    adapter_checkpoint: str,
    tokenizer_dir: str,
    vae_path: str,
    output_dir: str,
    mask_image: str | None,
    use_color_reference: bool,
    color_reference: str | None,
    image_size: int,
    sampling_steps: int,
    cfg_scale: float,
    mask_scale: float,
    seed: int,
    samples: int,
    precision: str,
    gpu_ids: str,
):
    language = "zh" if language_choice == LANG_TEXTS["zh"]["language_zh"] else DEFAULT_LANGUAGE
    texts = get_texts(language)

    prompt = prompt.strip()
    if not prompt:
        raise gr.Error(texts["error_caption_empty"])

    gpu_ids = gpu_ids.strip()
    if not gpu_ids:
        raise gr.Error(texts["error_gpu_empty"])

    base_ckpt = resolve_ckpt_file(base_checkpoint_dir, "consolidated.00-of-01.pth", texts["base_ckpt"])
    mask_path = mask_image if mask_image else None
    adapter_value = (adapter_checkpoint or "").strip()
    adapter_ckpt = resolve_ckpt_file(adapter_value, "adapter.pth", texts["adapter_ckpt"]) if adapter_value and mask_path else None
    tokenizer_path = resolve_required_dir(tokenizer_dir, texts["tokenizer"])
    resolved_vae = optional_existing_path(vae_path, texts["vae"])
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    ref_path = color_reference if (use_color_reference and color_reference) else None
    effective_prompt = strip_color_style_prompt(prompt) if ref_path else prompt
    prompt_processing = texts["summary_struct_hints_disabled"]
    if mask_path:
        effective_prompt = augment_caption_with_struct_hints(
            effective_prompt,
            image_path=None,
            mask_path=mask_path,
            struct_mask_channels=DEFAULT_STRUCT_MASK_CHANNELS,
        )
        prompt_processing = texts["summary_struct_hints_enabled"]
    effective_adapter_ckpt = adapter_ckpt if mask_path else None
    image_paths = []
    all_logs = []
    base_seed = int(seed)
    for idx in range(int(samples)):
        image_path, logs = run_single_sample(
            prompt=effective_prompt,
            base_ckpt=base_ckpt,
            adapter_ckpt=effective_adapter_ckpt,
            tokenizer_dir=tokenizer_path,
            vae_path=resolved_vae,
            output_dir=output_path,
            mask_path=mask_path,
            color_reference=ref_path,
            texts=texts,
            image_size=int(image_size),
            sampling_steps=int(sampling_steps),
            cfg_scale=float(cfg_scale),
            mask_scale=float(mask_scale),
            seed=base_seed + idx,
            precision=precision,
            index=idx,
            gpu_ids=gpu_ids,
        )
        image_paths.append(image_path)
        all_logs.append(f"===== sample {idx + 1} / {samples} =====\n{logs}")

    summary = (
        f"{texts['summary_final_run']}: {FINAL_RUN_NAME}\n"
        f"{texts['summary_base_ckpt']}: {base_ckpt}\n"
        f"{texts['summary_adapter_ckpt']}: {effective_adapter_ckpt or texts['no_mask']}\n"
        f"{texts['summary_mode']}: {texts['summary_mode_mm'] if effective_adapter_ckpt is not None else texts['summary_mode_base']}\n"
        f"{texts['summary_tokenizer']}: {tokenizer_path}\n"
        f"{texts['summary_vae']}: {resolved_vae or 'diffusers default'}\n"
        f"{texts['summary_mask']}: {mask_path or texts['no_mask']}\n"
        f"{texts['summary_color_reference']}: {ref_path or texts['no_color_reference']}\n"
        f"{texts['summary_prompt_processing']}: {prompt_processing}\n"
        f"Struct mask channels: {DEFAULT_STRUCT_MASK_CHANNELS}\n"
        f"{texts['summary_image_size']}: {image_size}\n"
        f"{texts['summary_steps']}: {sampling_steps}\n"
        f"{texts['summary_cfg']}: {cfg_scale}\n"
        f"{texts['summary_mask_scale']}: {mask_scale}\n"
        f"{texts['summary_seed_range']}: {base_seed}-{base_seed + int(samples) - 1}\n"
        f"{texts['summary_precision']}: {precision}\n"
        f"{texts['summary_gpu']}: {gpu_ids}"
    )
    return image_paths, summary, "\n\n".join(all_logs)


def build_interface():
    texts = get_texts(DEFAULT_LANGUAGE)
    with gr.Blocks(title="Stage A 1K Final Generator", css=APP_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            with gr.Column(elem_classes=["hero-panel"]):
                with gr.Row(equal_height=False, elem_classes=["hero-layout"]):
                    with gr.Column(scale=8):
                        hero = gr.HTML(hero_html(DEFAULT_LANGUAGE))
                    with gr.Column(scale=2, elem_classes=["hero-lang"]):
                        language_choice = gr.Dropdown(
                            label="Language / 语言",
                            choices=[texts["language_en"], texts["language_zh"]],
                            value=texts["language_en"],
                            show_label=False,
                        )

            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["control-panel"]):
                    prompt = gr.Textbox(
                        label=texts["prompt"],
                        value="",
                        placeholder=texts["prompt_placeholder"],
                        lines=5,
                    )
                    with gr.Accordion(texts["model"], open=True) as model_accordion:
                        base_checkpoint_dir = gr.Textbox(
                            label=texts["base_ckpt"],
                            value=existing_value(DEFAULT_BASE_CHECKPOINT_DIR),
                        )
                        adapter_checkpoint = gr.Textbox(
                            label=texts["adapter_ckpt"],
                            value=existing_value(DEFAULT_FINAL_ADAPTER),
                        )
                        tokenizer_dir = gr.Textbox(
                            label=texts["tokenizer"],
                            value=existing_value(DEFAULT_TOKENIZER_DIR),
                        )
                        vae_path = gr.Textbox(
                            label=texts["vae"],
                            value=existing_value(DEFAULT_VAE_PATH),
                        )
                        output_dir = gr.Textbox(
                            label=texts["output"],
                            value=existing_value(DEFAULT_OUTPUT_DIR),
                        )

                    with gr.Accordion(texts["conditions"], open=True) as conditions_accordion:
                        mask_image = gr.Image(
                            label=texts["mask"],
                            value=None,
                            type="filepath",
                            image_mode="RGB",
                        )
                        use_color_reference = gr.Checkbox(
                            label=texts["use_color_reference"],
                            value=False,
                        )
                        with gr.Column(visible=False) as color_reference_box:
                            color_reference = gr.Image(
                                label=texts["color_reference"],
                                type="filepath",
                                image_mode="RGB",
                            )

                    with gr.Accordion(texts["settings"], open=True) as settings_accordion:
                        with gr.Row():
                            image_size = gr.Slider(label=texts["image_size"], minimum=256, maximum=768, value=512, step=64)
                            precision = gr.Dropdown(label=texts["precision"], choices=["bf16", "fp16", "fp32"], value="bf16")
                        with gr.Row():
                            sampling_steps = gr.Slider(label=texts["steps"], minimum=20, maximum=160, value=80, step=5)
                            cfg_scale = gr.Slider(label=texts["cfg"], minimum=1.0, maximum=6.0, value=2.0, step=0.1)
                            mask_scale = gr.Slider(label=texts["mask_scale"], minimum=0.0, maximum=4.0, value=1.0, step=0.1)
                        with gr.Row():
                            seed = gr.Number(label=texts["seed"], value=42, precision=0)
                            samples = gr.Slider(label=texts["samples"], minimum=1, maximum=4, value=1, step=1)
                            gpu_ids = gr.Textbox(label=texts["gpu"], value="0")

                    run_button = gr.Button(texts["run"], variant="primary", elem_classes=["run-button"])

                with gr.Column(scale=7, elem_classes=["result-panel"]):
                    gallery = gr.Gallery(label=texts["gallery"], columns=2, height=430)
                    summary = gr.Textbox(label=texts["summary"], lines=12)
                    logs = gr.Textbox(label=texts["logs"], lines=16)

            language_choice.change(
                fn=language_updates,
                inputs=[language_choice],
                outputs=[
                    hero,
                    prompt,
                    model_accordion,
                    base_checkpoint_dir,
                    adapter_checkpoint,
                    tokenizer_dir,
                    vae_path,
                    output_dir,
                    conditions_accordion,
                    mask_image,
                    use_color_reference,
                    color_reference,
                    settings_accordion,
                    image_size,
                    precision,
                    sampling_steps,
                    cfg_scale,
                    mask_scale,
                    seed,
                    samples,
                    gpu_ids,
                    run_button,
                    gallery,
                    summary,
                    logs,
                ],
            )

            use_color_reference.change(
                fn=lambda enabled: gr.update(visible=bool(enabled)),
                inputs=[use_color_reference],
                outputs=[color_reference_box],
            )

            run_button.click(
                fn=run_generation,
                inputs=[
                    language_choice,
                    prompt,
                    base_checkpoint_dir,
                    adapter_checkpoint,
                    tokenizer_dir,
                    vae_path,
                    output_dir,
                    mask_image,
                    use_color_reference,
                    color_reference,
                    image_size,
                    sampling_steps,
                    cfg_scale,
                    mask_scale,
                    seed,
                    samples,
                    precision,
                    gpu_ids,
                ],
                outputs=[gallery, summary, logs],
            )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio UI for the selected Stage A 1K RetinaLogos adapter.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_interface().launch(server_name=args.host, server_port=args.port, share=args.share)
