import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFilter


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "inference_results"
DEFAULT_CHECKPOINT_DIR = PROJECT_DIR.parent / "checkpoints"
DEFAULT_TOKENIZER_DIR = PROJECT_DIR / "google_gemma-2b"
DEFAULT_VAE_DIR = PROJECT_DIR / "sdxl-vae"

TEXTS = {
    "zh": {
        "lang_label": "Language / \u8bed\u8a00",
        "lang_cn": "\u4e2d\u6587",
        "lang_en": "English",
        "title": "RetinaLogos \u773c\u5e95\u56fe\u751f\u6210\u6f14\u793a\u53f0",
        "subtitle": "\u5f53\u524d\u53ea\u652f\u6301\u6587\u672c\u751f\u6210\u773c\u5e95\u56fe\u3002Mask \u5f00\u5173\u4ec5\u4f5c\u9884\u7559\uff0c\u4e0d\u53c2\u4e0e\u5b9e\u9645\u63a8\u7406\u3002",
        "badge_stage": "\u9636\u6bb5\uff1a\u6587\u672c\u751f\u56fe\u57fa\u7ebf",
        "badge_backend": "\u540e\u7aef\uff1asample_from_seeds_multigpus.py",
        "badge_mode": "\u6a21\u5f0f\uff1a\u5355\u5361\u9a8c\u8bc1",
        "offline_tip": "\u79bb\u7ebf\u670d\u52a1\u5668\u8bf7\u586b\u5199\u672c\u5730 VAE \u8def\u5f84\uff0c\u5426\u5219\u7a0b\u5e8f\u4f1a\u5c1d\u8bd5\u4ece\u9ed8\u8ba4\u7f13\u5b58\u6216 Hugging Face \u52a0\u8f7d\u3002",
        "prompt_label": "Prompt / \u63d0\u793a\u8bcd",
        "prompt_placeholder": "\u8f93\u5165\u82f1\u6587\u533b\u5b66\u63cf\u8ff0\uff0c\u7528\u4e8e\u751f\u6210\u773c\u5e95\u56fe\u3002",
        "paths": "Paths / \u8def\u5f84",
        "ckpt_label": "Checkpoint Directory / \u6743\u91cd\u76ee\u5f55",
        "tokenizer_label": "Tokenizer Directory / Tokenizer \u76ee\u5f55",
        "vae_label": "Local VAE Path / \u672c\u5730 VAE \u8def\u5f84",
        "vae_placeholder": "\u53ef\u9009\uff0c\u4f8b\u5982 /home/maziheng/models/sdxl-vae",
        "output_label": "Output Directory / \u8f93\u51fa\u76ee\u5f55",
        "settings": "Inference Settings / \u63a8\u7406\u53c2\u6570",
        "resolution_label": "Resolution / \u5206\u8fa8\u7387",
        "precision_label": "Precision / \u7cbe\u5ea6",
        "ema_label": "Use EMA / \u4f7f\u7528 EMA",
        "steps_label": "Sampling Steps / \u91c7\u6837\u6b65\u6570",
        "cfg_label": "CFG Scale / CFG \u7cfb\u6570",
        "seed_label": "Seed / \u968f\u673a\u79cd\u5b50",
        "samples_label": "Samples per Prompt / \u6bcf\u6761\u63d0\u793a\u751f\u6210\u5f20\u6570",
        "gpu_label": "CUDA_VISIBLE_DEVICES",
        "mask_group": "Reserved Multimodal Entry / \u591a\u6a21\u6001\u9884\u7559\u5165\u53e3",
        "mask_toggle": "Enable Structure Mask (Reserved) / \u542f\u7528\u7ed3\u6784 Mask\uff08\u9884\u7559\uff09",
        "mask_off": "\u5f53\u524d\u672a\u542f\u7528 Mask \u9884\u7559\u5165\u53e3\uff0c\u9875\u9762\u4ecd\u6309\u7eaf\u6587\u672c\u751f\u56fe\u6267\u884c\u3002",
        "mask_on": "\u5df2\u6253\u5f00 Mask \u9884\u7559\u5165\u53e3\uff0c\u4f46\u5f53\u524d\u540e\u7aef\u4ecd\u5ffd\u7565 Mask\uff0c\u4ec5\u7528\u4e8e\u5c55\u793a\u540e\u7eed\u6269\u5c55\u65b9\u5411\u3002",
        "cleanup_group": "Demo Cleanup / \u6f14\u793a\u540e\u5904\u7406",
        "cleanup_toggle": "Apply ideal circular aperture / \u5957\u7528\u7406\u60f3\u5706\u5f62\u5b54\u5f84",
        "cleanup_note_on": "\u5df2\u542f\u7528\u6f14\u793a\u540e\u5904\u7406\uff0c\u5c06\u5bf9\u5c55\u793a\u56fe\u50cf\u7684\u5706\u5f62\u89c6\u91ce\u5916\u533a\u57df\u8fdb\u884c\u6807\u51c6\u5316\u906e\u7f69\u3002",
        "cleanup_note_off": "\u5df2\u5173\u95ed\u6f14\u793a\u540e\u5904\u7406\uff0c\u76f4\u63a5\u663e\u793a\u539f\u59cb\u751f\u6210\u56fe\u50cf\u3002",
        "run_button": "\u5f00\u59cb\u63a8\u7406",
        "gallery_label": "\u751f\u6210\u7ed3\u679c",
        "summary_label": "\u8fd0\u884c\u6458\u8981",
        "logs_label": "\u547d\u4ee4\u65e5\u5fd7",
        "prompt_required": "Prompt \u4e0d\u80fd\u4e3a\u7a7a\u3002",
        "checkpoint_missing": "Checkpoint \u76ee\u5f55\u4e0d\u5b58\u5728",
        "tokenizer_missing": "Tokenizer \u76ee\u5f55\u4e0d\u5b58\u5728",
        "raw_checkpoint_missing": "\u672a\u627e\u5230\u57fa\u7840\u6743\u91cd\u6587\u4ef6",
        "ema_checkpoint_missing": "\u52fe\u9009\u4e86 EMA\uff0c\u4f46\u672a\u627e\u5230 consolidated_ema.00-of-01.pth",
        "vae_path_missing": "\u672c\u5730 VAE \u8def\u5f84\u4e0d\u5b58\u5728",
        "vae_config_missing": "\u672c\u5730 VAE \u8def\u5f84\u4e0b\u672a\u627e\u5230 config.json",
        "gpu_required": "CUDA_VISIBLE_DEVICES \u4e0d\u80fd\u4e3a\u7a7a\uff0c\u4f8b\u5982 0",
        "resolution_format": "\u5206\u8fa8\u7387\u683c\u5f0f\u5e94\u4e3a 512x512",
        "resolution_numeric": "\u5206\u8fa8\u7387\u5fc5\u987b\u4e3a\u6570\u5b57\uff0c\u4f8b\u5982 512x512",
        "run_dir_missing": "\u63a8\u7406\u7ed3\u675f\uff0c\u4f46\u6ca1\u6709\u627e\u5230\u8f93\u51fa\u76ee\u5f55",
        "image_missing": "\u63a8\u7406\u7ed3\u675f\uff0c\u4f46\u6ca1\u6709\u751f\u6210\u56fe\u50cf",
        "error_title": "\u63a8\u7406\u5931\u8d25\u3002",
        "summary_template": (
            "Checkpoint: {ckpt}\n"
            "Tokenizer: {tokenizer}\n"
            "VAE Path: {vae}\n"
            "Resolution: {resolution}\n"
            "Sampling Steps: {steps}\n"
            "CFG Scale: {cfg}\n"
            "Seed: {seed}\n"
            "Samples per Prompt: {samples}\n"
            "EMA: {ema}\n"
            "Mask Reserve: {mask}\n"
            "Demo Aperture Cleanup: {cleanup}\n"
            "Visible GPU IDs: {gpu}"
        ),
    },
    "en": {
        "lang_label": "Language / \u8bed\u8a00",
        "lang_cn": "\u4e2d\u6587",
        "lang_en": "English",
        "title": "RetinaLogos Medical Image Generation Console",
        "subtitle": "Current phase: text-to-fundus only. The mask switch is reserved for future work and does not affect inference.",
        "badge_stage": "Stage: text baseline",
        "badge_backend": "Backend: sample_from_seeds_multigpus.py",
        "badge_mode": "Mode: single-GPU validation",
        "offline_tip": "For offline servers, fill in a local VAE path. Otherwise the script will try the default cache or Hugging Face.",
        "prompt_label": "Prompt / \u63d0\u793a\u8bcd",
        "prompt_placeholder": "Enter an English medical description for retinal image generation.",
        "paths": "Paths / \u8def\u5f84",
        "ckpt_label": "Checkpoint Directory / \u6743\u91cd\u76ee\u5f55",
        "tokenizer_label": "Tokenizer Directory / Tokenizer \u76ee\u5f55",
        "vae_label": "Local VAE Path / \u672c\u5730 VAE \u8def\u5f84",
        "vae_placeholder": "Optional, for example /home/maziheng/models/sdxl-vae",
        "output_label": "Output Directory / \u8f93\u51fa\u76ee\u5f55",
        "settings": "Inference Settings / \u63a8\u7406\u53c2\u6570",
        "resolution_label": "Resolution / \u5206\u8fa8\u7387",
        "precision_label": "Precision / \u7cbe\u5ea6",
        "ema_label": "Use EMA / \u4f7f\u7528 EMA",
        "steps_label": "Sampling Steps / \u91c7\u6837\u6b65\u6570",
        "cfg_label": "CFG Scale / CFG \u7cfb\u6570",
        "seed_label": "Seed / \u968f\u673a\u79cd\u5b50",
        "samples_label": "Samples per Prompt / \u6bcf\u6761\u63d0\u793a\u751f\u6210\u5f20\u6570",
        "gpu_label": "CUDA_VISIBLE_DEVICES",
        "mask_group": "Reserved Multimodal Entry / \u591a\u6a21\u6001\u9884\u7559\u5165\u53e3",
        "mask_toggle": "Enable Structure Mask (Reserved) / \u542f\u7528\u7ed3\u6784 Mask\uff08\u9884\u7559\uff09",
        "mask_off": "Mask reserve is off. The page runs pure text-to-image inference.",
        "mask_on": "Mask reserve is on, but the current backend still ignores mask input. This is UI-only for future extension.",
        "cleanup_group": "Demo Cleanup / \u6f14\u793a\u540e\u5904\u7406",
        "cleanup_toggle": "Apply ideal circular aperture / \u5957\u7528\u7406\u60f3\u5706\u5f62\u5b54\u5f84",
        "cleanup_note_on": "Demo cleanup is enabled. Display images will be masked to an ideal circular fundus aperture.",
        "cleanup_note_off": "Demo cleanup is disabled. Raw generated images will be shown directly.",
        "run_button": "Run Inference",
        "gallery_label": "Generated Images",
        "summary_label": "Run Summary",
        "logs_label": "Command Logs",
        "prompt_required": "Prompt is required.",
        "checkpoint_missing": "Checkpoint directory not found",
        "tokenizer_missing": "Tokenizer directory not found",
        "raw_checkpoint_missing": "Base checkpoint file not found",
        "ema_checkpoint_missing": "EMA was selected, but consolidated_ema.00-of-01.pth was not found",
        "vae_path_missing": "Local VAE path not found",
        "vae_config_missing": "config.json was not found under the local VAE path",
        "gpu_required": "CUDA_VISIBLE_DEVICES cannot be empty. Example: 0",
        "resolution_format": "Resolution must look like 512x512.",
        "resolution_numeric": "Resolution must be numeric, for example 512x512.",
        "run_dir_missing": "Run finished, but no output folder was found",
        "image_missing": "Run finished, but no images were generated",
        "error_title": "Inference failed.",
        "summary_template": (
            "Checkpoint: {ckpt}\n"
            "Tokenizer: {tokenizer}\n"
            "VAE Path: {vae}\n"
            "Resolution: {resolution}\n"
            "Sampling Steps: {steps}\n"
            "CFG Scale: {cfg}\n"
            "Seed: {seed}\n"
            "Samples per Prompt: {samples}\n"
            "EMA: {ema}\n"
            "Mask Reserve: {mask}\n"
            "Demo Aperture Cleanup: {cleanup}\n"
            "Visible GPU IDs: {gpu}"
        ),
    },
}

APP_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(34, 122, 103, 0.12), transparent 28%),
    radial-gradient(circle at top right, rgba(198, 114, 54, 0.12), transparent 24%),
    linear-gradient(180deg, #f7f4ef 0%, #f2ede3 100%);
  color: #1f2a2e;
}
.app-shell {
  max-width: 1380px;
  margin: 0 auto;
}
.hero-panel,
.control-panel,
.result-panel {
  border-radius: 26px;
  border: 1px solid rgba(37, 81, 72, 0.12);
  box-shadow: 0 18px 60px rgba(55, 74, 67, 0.08);
  backdrop-filter: blur(8px);
}
.hero-panel {
  background: linear-gradient(135deg, rgba(14, 91, 82, 0.96), rgba(30, 58, 95, 0.92));
  color: #f9faf8;
  padding: 8px 10px 2px 10px;
}
.hero-copy {
  text-align: center;
  padding: 18px 18px 14px 18px;
}
.hero-copy .title {
  font-size: 33px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #f9faf8;
}
.hero-copy .subtitle {
  margin-top: 10px;
  max-width: 980px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.75;
  font-size: 15px;
  color: #f9faf8;
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  margin-top: 14px;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.16);
  font-size: 13px;
  color: #f9faf8;
}
.control-panel {
  background: rgba(255, 252, 247, 0.92);
  padding: 12px;
}
.result-panel {
  background: rgba(249, 250, 248, 0.94);
  padding: 12px;
}
.section-note {
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(31, 112, 102, 0.08), rgba(198, 114, 54, 0.08));
  border: 1px solid rgba(37, 81, 72, 0.12);
  padding: 10px 14px;
  margin-bottom: 10px;
  font-size: 14px;
}
.reserve-note {
  border-radius: 18px;
  background: rgba(255, 244, 230, 0.95);
  border: 1px solid rgba(198, 114, 54, 0.18);
  padding: 10px 14px;
  margin-top: 6px;
  font-size: 14px;
}
.cleanup-note {
  border-radius: 18px;
  background: rgba(231, 245, 243, 0.92);
  border: 1px solid rgba(31, 123, 112, 0.18);
  padding: 10px 14px;
  margin-top: 6px;
  font-size: 14px;
}
.run-button button {
  height: 56px;
  border: none !important;
  border-radius: 18px !important;
  background: linear-gradient(135deg, #1f7b70, #c67236) !important;
  color: #fff !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em;
}
"""


def get_lang_code(choice: str) -> str:
    return "zh" if choice == "CN" else "en"


def t(choice: str) -> dict:
    return TEXTS[get_lang_code(choice)]


def hero_html(choice: str) -> str:
    text = t(choice)
    return (
        '<div class="hero-copy">'
        f'<div class="title">{text["title"]}</div>'
        f'<div class="subtitle">{text["subtitle"]}</div>'
        '<div class="badge-row">'
        f'<span class="badge">{text["badge_stage"]}</span>'
        f'<span class="badge">{text["badge_backend"]}</span>'
        f'<span class="badge">{text["badge_mode"]}</span>'
        "</div></div>"
    )


def offline_tip_html(choice: str) -> str:
    return f'<div class="section-note">{t(choice)["offline_tip"]}</div>'


def mask_note_html(choice: str, reserve_mask: bool) -> str:
    text = t(choice)
    return f'<div class="reserve-note">{text["mask_on"] if reserve_mask else text["mask_off"]}</div>'


def cleanup_note_html(choice: str, cleanup_aperture: bool) -> str:
    text = t(choice)
    return (
        f'<div class="cleanup-note">'
        f'{text["cleanup_note_on"] if cleanup_aperture else text["cleanup_note_off"]}'
        f"</div>"
    )


def normalize_resolution(resolution: str, choice: str) -> str:
    text = t(choice)
    value = resolution.strip().lower().replace(" ", "")
    if "x" not in value:
        raise gr.Error(text["resolution_format"])
    width, height = value.split("x", 1)
    if not width.isdigit() or not height.isdigit():
        raise gr.Error(text["resolution_numeric"])
    return f"{width}:{width}x{height}"


def find_latest_images(run_dir: Path) -> list[str]:
    image_dir = run_dir / "images"
    if not image_dir.exists():
        return []
    return [str(path) for path in sorted(image_dir.glob("*.png"))]


def detect_new_run(output_dir: Path, before: set[str]) -> Path | None:
    after = {path.name for path in output_dir.iterdir() if path.is_dir()} if output_dir.exists() else set()
    new_dirs = sorted(after - before)
    if new_dirs:
        return output_dir / new_dirs[-1]
    if not output_dir.exists():
        return None
    existing = sorted([path for path in output_dir.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime)
    return existing[-1] if existing else None


def build_summary(
    choice: str,
    ckpt_path: Path,
    tokenizer_path: Path,
    vae_path: str,
    resolution: str,
    sampling_steps: int,
    cfg_scale: float,
    seed: int,
    samples_per_caption: int,
    use_ema: bool,
    reserve_mask: bool,
    cleanup_aperture: bool,
    gpu_ids: str,
) -> str:
    return t(choice)["summary_template"].format(
        ckpt=ckpt_path,
        tokenizer=tokenizer_path,
        vae=vae_path or "AUTO",
        resolution=resolution,
        steps=sampling_steps,
        cfg=cfg_scale,
        seed=int(seed),
        samples=samples_per_caption,
        ema=use_ema,
        mask="RESERVED" if reserve_mask else "OFF",
        cleanup=cleanup_aperture,
        gpu=gpu_ids,
    )


def apply_demo_aperture(image_path: str, target_dir: Path) -> str:
    source = Path(image_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / source.name

    with Image.open(source).convert("RGBA") as image:
        width, height = image.size
        margin = max(8, int(min(width, height) * 0.018))
        feather = max(2, int(min(width, height) * 0.01))

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (
                margin,
                margin,
                width - margin,
                height - margin,
            ),
            fill=255,
        )
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

        composited = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        composited.paste(image, (0, 0), mask)
        composited.convert("RGB").save(output_path)

    return str(output_path)


def update_ui(choice: str, reserve_mask: bool, cleanup_aperture: bool):
    text = t(choice)
    return (
        hero_html(choice),
        offline_tip_html(choice),
        gr.update(label=text["prompt_label"], placeholder=text["prompt_placeholder"]),
        gr.update(label=text["paths"]),
        gr.update(label=text["ckpt_label"]),
        gr.update(label=text["tokenizer_label"]),
        gr.update(label=text["vae_label"], placeholder=text["vae_placeholder"]),
        gr.update(label=text["output_label"]),
        gr.update(label=text["settings"]),
        gr.update(label=text["resolution_label"]),
        gr.update(label=text["precision_label"]),
        gr.update(label=text["ema_label"]),
        gr.update(label=text["steps_label"]),
        gr.update(label=text["cfg_label"]),
        gr.update(label=text["seed_label"]),
        gr.update(label=text["samples_label"]),
        gr.update(label=text["gpu_label"]),
        gr.update(label=text["mask_group"]),
        gr.update(label=text["mask_toggle"]),
        mask_note_html(choice, reserve_mask),
        gr.update(label=text["cleanup_group"]),
        gr.update(label=text["cleanup_toggle"]),
        cleanup_note_html(choice, cleanup_aperture),
        gr.update(value=text["run_button"]),
        gr.update(label=text["gallery_label"]),
        gr.update(label=text["summary_label"]),
        gr.update(label=text["logs_label"]),
    )


def update_mask_note(choice: str, reserve_mask: bool):
    return mask_note_html(choice, reserve_mask)


def update_cleanup_note(choice: str, cleanup_aperture: bool):
    return cleanup_note_html(choice, cleanup_aperture)


def run_generation(
    choice: str,
    prompt: str,
    checkpoint_dir: str,
    tokenizer_dir: str,
    local_vae_path: str,
    output_dir: str,
    reserve_mask: bool,
    cleanup_aperture: bool,
    resolution: str,
    sampling_steps: int,
    cfg_scale: float,
    seed: int,
    samples_per_caption: int,
    gpu_ids: str,
    precision: str,
    use_ema: bool,
):
    text = t(choice)
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error(text["prompt_required"])

    ckpt_path = Path(checkpoint_dir).expanduser().resolve()
    if not ckpt_path.exists():
        raise gr.Error(f'{text["checkpoint_missing"]}: {ckpt_path}')

    tokenizer_path = Path(tokenizer_dir).expanduser().resolve()
    if not tokenizer_path.exists():
        raise gr.Error(f'{text["tokenizer_missing"]}: {tokenizer_path}')

    raw_ckpt = ckpt_path / "consolidated.00-of-01.pth"
    ema_ckpt = ckpt_path / "consolidated_ema.00-of-01.pth"
    if use_ema and not ema_ckpt.exists():
        raise gr.Error(f'{text["ema_checkpoint_missing"]}: {ema_ckpt}')
    if not use_ema and not raw_ckpt.exists():
        raise gr.Error(f'{text["raw_checkpoint_missing"]}: {raw_ckpt}')

    vae_path = local_vae_path.strip()
    if vae_path:
        vae_dir = Path(vae_path).expanduser().resolve()
        if not vae_dir.exists():
            raise gr.Error(f'{text["vae_path_missing"]}: {vae_dir}')
        if not (vae_dir / "config.json").exists():
            raise gr.Error(f'{text["vae_config_missing"]}: {vae_dir}')
        vae_path = str(vae_dir)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    existing_runs = {path.name for path in output_path.iterdir() if path.is_dir()}

    resolution_arg = normalize_resolution(resolution, choice)
    gpu_ids = gpu_ids.strip()
    if not gpu_ids:
        raise gr.Error(text["gpu_required"])

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(prompt + "\n")
        caption_file = Path(temp_file.name)

    command = [
        sys.executable,
        "sample_from_seeds_multigpus.py",
        "--id",
        "gradio_demo",
        "--ckpt",
        str(ckpt_path),
        "--image_save_path",
        str(output_path),
        "--caption_path",
        str(caption_file),
        "--resolution",
        resolution_arg,
        "--num_sampling_steps",
        str(sampling_steps),
        "--cfg_scale",
        str(cfg_scale),
        "--samples_per_caption",
        str(samples_per_caption),
        "--num_gpus",
        "1",
        "--precision",
        precision,
        "--seeds",
        str(int(seed)),
        "--tokenizer_path",
        str(tokenizer_path),
    ]
    if use_ema:
        command.append("--ema")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    if vae_path:
        env["RETINA_VAE_PATH"] = vae_path

    try:
        process = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        caption_file.unlink(missing_ok=True)

    logs = (process.stdout or "") + ("\n" + process.stderr if process.stderr else "")
    if process.returncode != 0:
        raise gr.Error(f'{text["error_title"]}\n\n{logs}')

    run_dir = detect_new_run(output_path, existing_runs)
    if run_dir is None:
        raise gr.Error(f'{text["run_dir_missing"]}\n\n{logs}')

    images = find_latest_images(run_dir)
    if not images:
        raise gr.Error(f'{text["image_missing"]}\n\n{logs}')

    if cleanup_aperture:
        cleaned_dir = run_dir / "demo_aperture"
        images = [apply_demo_aperture(path, cleaned_dir) for path in images]

    summary = build_summary(
        choice=choice,
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        vae_path=vae_path,
        resolution=resolution,
        sampling_steps=sampling_steps,
        cfg_scale=cfg_scale,
        seed=seed,
        samples_per_caption=samples_per_caption,
        use_ema=use_ema,
        reserve_mask=reserve_mask,
        cleanup_aperture=cleanup_aperture,
        gpu_ids=gpu_ids,
    )
    return images, summary, logs


def build_interface():
    with gr.Blocks(title="RetinaLogos Demo", css=APP_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            hero = gr.HTML(hero_html("EN"), elem_classes=["hero-panel"])

            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["control-panel"]):
                    choice = gr.Radio(
                        label=TEXTS["en"]["lang_label"],
                        choices=["EN", "CN"],
                        value="EN",
                    )

                    offline_tip = gr.HTML(offline_tip_html("EN"))

                    prompt = gr.Textbox(
                        label=TEXTS["en"]["prompt_label"],
                        lines=9,
                        placeholder=TEXTS["en"]["prompt_placeholder"],
                    )

                    with gr.Accordion(TEXTS["en"]["paths"], open=True) as paths_group:
                        checkpoint_dir = gr.Textbox(
                            label=TEXTS["en"]["ckpt_label"],
                            value=str(DEFAULT_CHECKPOINT_DIR.resolve()),
                        )
                        tokenizer_dir = gr.Textbox(
                            label=TEXTS["en"]["tokenizer_label"],
                            value=str(DEFAULT_TOKENIZER_DIR.resolve()),
                        )
                        local_vae_path = gr.Textbox(
                            label=TEXTS["en"]["vae_label"],
                            value=str(DEFAULT_VAE_DIR.resolve()),
                            placeholder=TEXTS["en"]["vae_placeholder"],
                        )
                        output_dir = gr.Textbox(
                            label=TEXTS["en"]["output_label"],
                            value=str(DEFAULT_OUTPUT_DIR.resolve()),
                        )

                    with gr.Accordion(TEXTS["en"]["settings"], open=True) as settings_group:
                        with gr.Row():
                            resolution = gr.Textbox(label=TEXTS["en"]["resolution_label"], value="512x512")
                            precision = gr.Dropdown(
                                label=TEXTS["en"]["precision_label"],
                                choices=["bf16", "fp16", "fp32", "tf32"],
                                value="bf16",
                            )
                            use_ema = gr.Checkbox(label=TEXTS["en"]["ema_label"], value=False)
                        with gr.Row():
                            sampling_steps = gr.Slider(
                                label=TEXTS["en"]["steps_label"],
                                minimum=20,
                                maximum=250,
                                value=50,
                                step=5,
                            )
                            cfg_scale = gr.Slider(
                                label=TEXTS["en"]["cfg_label"],
                                minimum=1.0,
                                maximum=8.0,
                                value=4.0,
                                step=0.5,
                            )
                        with gr.Row():
                            seed = gr.Number(label=TEXTS["en"]["seed_label"], value=42, precision=0)
                            samples_per_caption = gr.Slider(
                                label=TEXTS["en"]["samples_label"],
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                            )
                            gpu_ids = gr.Textbox(label=TEXTS["en"]["gpu_label"], value="0")

                    with gr.Accordion(TEXTS["en"]["mask_group"], open=False) as mask_group:
                        reserve_mask = gr.Checkbox(label=TEXTS["en"]["mask_toggle"], value=False)
                        mask_note = gr.HTML(mask_note_html("EN", False))

                    with gr.Accordion(TEXTS["en"]["cleanup_group"], open=False) as cleanup_group:
                        cleanup_aperture = gr.Checkbox(label=TEXTS["en"]["cleanup_toggle"], value=True)
                        cleanup_note = gr.HTML(cleanup_note_html("EN", True))

                    run_button = gr.Button(TEXTS["en"]["run_button"], variant="primary", elem_classes=["run-button"])

                with gr.Column(scale=7, elem_classes=["result-panel"]):
                    gallery = gr.Gallery(label=TEXTS["en"]["gallery_label"], columns=2, height=470)
                    summary = gr.Textbox(label=TEXTS["en"]["summary_label"], lines=10)
                    logs = gr.Textbox(label=TEXTS["en"]["logs_label"], lines=18)

            choice.change(
                fn=update_ui,
                inputs=[choice, reserve_mask, cleanup_aperture],
                outputs=[
                    hero,
                    offline_tip,
                    prompt,
                    paths_group,
                    checkpoint_dir,
                    tokenizer_dir,
                    local_vae_path,
                    output_dir,
                    settings_group,
                    resolution,
                    precision,
                    use_ema,
                    sampling_steps,
                    cfg_scale,
                    seed,
                    samples_per_caption,
                    gpu_ids,
                    mask_group,
                    reserve_mask,
                    mask_note,
                    cleanup_group,
                    cleanup_aperture,
                    cleanup_note,
                    run_button,
                    gallery,
                    summary,
                    logs,
                ],
            )

            reserve_mask.change(
                fn=update_mask_note,
                inputs=[choice, reserve_mask],
                outputs=[mask_note],
            )

            cleanup_aperture.change(
                fn=update_cleanup_note,
                inputs=[choice, cleanup_aperture],
                outputs=[cleanup_note],
            )

            run_button.click(
                fn=run_generation,
                inputs=[
                    choice,
                    prompt,
                    checkpoint_dir,
                    tokenizer_dir,
                    local_vae_path,
                    output_dir,
                    reserve_mask,
                    cleanup_aperture,
                    resolution,
                    sampling_steps,
                    cfg_scale,
                    seed,
                    samples_per_caption,
                    gpu_ids,
                    precision,
                    use_ema,
                ],
                outputs=[gallery, summary, logs],
            )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="ASCII-safe bilingual Gradio UI for RetinaLogos text-only inference.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_interface().launch(server_name=args.host, server_port=args.port, share=args.share)
