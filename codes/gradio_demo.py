import argparse
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent

DEFAULT_BASE_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
DEFAULT_FINAL_ADAPTER = REPO_ROOT / "checkpoints" / "stageA_1k_final" / "adapter.pth"
DEFAULT_TOKENIZER_DIR = PROJECT_DIR / "google_gemma-2b"
DEFAULT_VAE_PATH = PROJECT_DIR / "sdxl-vae"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "inference_stageA_1k_final"
DEFAULT_TRIPTYCH = REPO_ROOT / "checkpoints" / "stageA_1k_final" / "triptych_sheet_canonical.png"
DEFAULT_MASK = PROJECT_DIR / "data" / "train" / "diabetic" / "mask" / "10000_left_fusion.png"

FINAL_RUN_NAME = "mainline_stageA_global_1k_maskqc_clean_canonical_strictalign_colorstyle8_compact_frombase_colorfix"
FINAL_STEP = 1000

TEXT = {
    "title": "Stage A 1K Final Fundus Generator",
    "subtitle": (
        "Selected checkpoint: 0001000 adapter. Later 1.5K-5K continuation runs did not pass "
        "the structure and color guard, so this demo is fixed to the 1K final scheme."
    ),
    "base_ckpt": "Base checkpoint directory",
    "adapter_ckpt": "Selected 1K adapter",
    "tokenizer": "Tokenizer directory",
    "vae": "Local VAE path",
    "output": "Output directory",
    "prompt": "Caption",
    "mask": "Structure mask",
    "color_reference": "Color reference",
    "settings": "Sampling",
    "image_size": "Image size",
    "steps": "Sampling steps",
    "cfg": "CFG scale",
    "mask_scale": "Mask scale",
    "seed": "Seed",
    "samples": "Samples",
    "precision": "Precision",
    "gpu": "CUDA_VISIBLE_DEVICES",
    "run": "Generate",
    "preview": "Selected 1K evaluation triptych",
    "gallery": "Generated images",
    "summary": "Run summary",
    "logs": "Logs",
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
"""


def existing_value(path: Path) -> str:
    return str(path.resolve()) if path.exists() else str(path)


def hero_html() -> str:
    return (
        '<div class="hero-panel">'
        f'<div class="hero-title">{TEXT["title"]}</div>'
        f'<div class="hero-subtitle">{TEXT["subtitle"]}</div>'
        '<div class="badge-row">'
        f'<span class="badge">Final step: {FINAL_STEP}</span>'
        '<span class="badge warm">Colorfix 1K adapter selected</span>'
        '<span class="badge rose">1.5K-5K rejected by hard guard</span>'
        '<span class="badge">Backend: inference_mask.py</span>'
        "</div></div>"
    )


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
    adapter_ckpt: Path,
    tokenizer_dir: Path,
    vae_path: str,
    output_dir: Path,
    mask_path: str | None,
    color_reference: str | None,
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
        "--adapter_ckpt",
        str(adapter_ckpt),
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
        "--output_name",
        output_name,
    ]
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
        raise gr.Error(f"Inference failed for seed {seed}.\n\n{logs}")
    if not out_path.exists():
        raise gr.Error(f"Inference finished but no image was produced: {out_path}\n\n{logs}")
    return str(out_path), logs


def run_generation(
    prompt: str,
    base_checkpoint_dir: str,
    adapter_checkpoint: str,
    tokenizer_dir: str,
    vae_path: str,
    output_dir: str,
    mask_image: str | None,
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
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Caption cannot be empty.")

    gpu_ids = gpu_ids.strip()
    if not gpu_ids:
        raise gr.Error("CUDA_VISIBLE_DEVICES cannot be empty.")

    base_ckpt = resolve_ckpt_file(base_checkpoint_dir, "consolidated.00-of-01.pth", TEXT["base_ckpt"])
    adapter_ckpt = resolve_ckpt_file(adapter_checkpoint, "adapter.pth", TEXT["adapter_ckpt"])
    tokenizer_path = resolve_required_dir(tokenizer_dir, TEXT["tokenizer"])
    resolved_vae = optional_existing_path(vae_path, TEXT["vae"])
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    mask_path = mask_image if mask_image else None
    ref_path = color_reference if color_reference else None
    image_paths = []
    all_logs = []
    base_seed = int(seed)
    for idx in range(int(samples)):
        image_path, logs = run_single_sample(
            prompt=prompt,
            base_ckpt=base_ckpt,
            adapter_ckpt=adapter_ckpt,
            tokenizer_dir=tokenizer_path,
            vae_path=resolved_vae,
            output_dir=output_path,
            mask_path=mask_path,
            color_reference=ref_path,
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
        f"Final run: {FINAL_RUN_NAME}\n"
        f"Selected step: {FINAL_STEP}\n"
        f"Base checkpoint: {base_ckpt}\n"
        f"Adapter checkpoint: {adapter_ckpt}\n"
        f"Tokenizer: {tokenizer_path}\n"
        f"VAE: {resolved_vae or 'diffusers default'}\n"
        f"Mask: {mask_path or 'disabled'}\n"
        f"Color reference: {ref_path or 'disabled'}\n"
        f"Image size: {image_size}\n"
        f"Sampling steps: {sampling_steps}\n"
        f"CFG scale: {cfg_scale}\n"
        f"Mask scale: {mask_scale}\n"
        f"Seed range: {base_seed}-{base_seed + int(samples) - 1}\n"
        f"Precision: {precision}\n"
        f"GPU IDs: {gpu_ids}"
    )
    return image_paths, summary, "\n\n".join(all_logs)


def build_interface():
    preview_value = str(DEFAULT_TRIPTYCH.resolve()) if DEFAULT_TRIPTYCH.exists() else None
    mask_value = str(DEFAULT_MASK.resolve()) if DEFAULT_MASK.exists() else None

    with gr.Blocks(title="Stage A 1K Final Generator", css=APP_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(hero_html())

            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["control-panel"]):
                    prompt = gr.Textbox(
                        label=TEXT["prompt"],
                        value=(
                            "A color fundus photograph with clear optic disc, macula, retinal vessels, "
                            "and mild diabetic retinopathy signs."
                        ),
                        lines=5,
                    )
                    with gr.Accordion("Model", open=True):
                        base_checkpoint_dir = gr.Textbox(
                            label=TEXT["base_ckpt"],
                            value=existing_value(DEFAULT_BASE_CHECKPOINT_DIR),
                        )
                        adapter_checkpoint = gr.Textbox(
                            label=TEXT["adapter_ckpt"],
                            value=existing_value(DEFAULT_FINAL_ADAPTER),
                        )
                        tokenizer_dir = gr.Textbox(
                            label=TEXT["tokenizer"],
                            value=existing_value(DEFAULT_TOKENIZER_DIR),
                        )
                        vae_path = gr.Textbox(
                            label=TEXT["vae"],
                            value=existing_value(DEFAULT_VAE_PATH),
                        )
                        output_dir = gr.Textbox(
                            label=TEXT["output"],
                            value=existing_value(DEFAULT_OUTPUT_DIR),
                        )

                    with gr.Accordion("Conditions", open=True):
                        mask_image = gr.Image(
                            label=TEXT["mask"],
                            value=mask_value,
                            type="filepath",
                            image_mode="L",
                        )
                        color_reference = gr.Image(
                            label=TEXT["color_reference"],
                            type="filepath",
                            image_mode="RGB",
                        )

                    with gr.Accordion(TEXT["settings"], open=True):
                        with gr.Row():
                            image_size = gr.Slider(label=TEXT["image_size"], minimum=256, maximum=768, value=512, step=64)
                            precision = gr.Dropdown(label=TEXT["precision"], choices=["bf16", "fp16", "fp32"], value="bf16")
                        with gr.Row():
                            sampling_steps = gr.Slider(label=TEXT["steps"], minimum=20, maximum=160, value=80, step=5)
                            cfg_scale = gr.Slider(label=TEXT["cfg"], minimum=1.0, maximum=6.0, value=2.0, step=0.1)
                            mask_scale = gr.Slider(label=TEXT["mask_scale"], minimum=0.0, maximum=4.0, value=1.0, step=0.1)
                        with gr.Row():
                            seed = gr.Number(label=TEXT["seed"], value=42, precision=0)
                            samples = gr.Slider(label=TEXT["samples"], minimum=1, maximum=4, value=1, step=1)
                            gpu_ids = gr.Textbox(label=TEXT["gpu"], value="0")

                    run_button = gr.Button(TEXT["run"], variant="primary", elem_classes=["run-button"])

                with gr.Column(scale=7, elem_classes=["result-panel"]):
                    gr.Image(value=preview_value, label=TEXT["preview"], type="filepath", height=300)
                    gallery = gr.Gallery(label=TEXT["gallery"], columns=2, height=430)
                    summary = gr.Textbox(label=TEXT["summary"], lines=12)
                    logs = gr.Textbox(label=TEXT["logs"], lines=16)

            run_button.click(
                fn=run_generation,
                inputs=[
                    prompt,
                    base_checkpoint_dir,
                    adapter_checkpoint,
                    tokenizer_dir,
                    vae_path,
                    output_dir,
                    mask_image,
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
