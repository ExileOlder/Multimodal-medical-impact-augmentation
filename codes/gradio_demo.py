import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "inference_results"
DEFAULT_CHECKPOINT_DIR = PROJECT_DIR.parent / "checkpoints"
DEFAULT_TOKENIZER_DIR = PROJECT_DIR / "google_gemma-2b"


def _normalize_resolution(resolution: str) -> str:
    value = resolution.strip().lower().replace(" ", "")
    if "x" not in value:
        raise gr.Error("Resolution must look like 512x512.")
    width, height = value.split("x", 1)
    if not width.isdigit() or not height.isdigit():
        raise gr.Error("Resolution must be numeric, for example 512x512.")
    return f"{width}:{width}x{height}"


def _find_latest_images(run_dir: Path) -> list[str]:
    image_dir = run_dir / "images"
    if not image_dir.exists():
        return []
    return [str(path) for path in sorted(image_dir.glob("*.png"))]


def _detect_new_run(output_dir: Path, before: set[str]) -> Path | None:
    after = {path.name for path in output_dir.iterdir() if path.is_dir()} if output_dir.exists() else set()
    new_dirs = sorted(after - before)
    if new_dirs:
        return output_dir / new_dirs[-1]
    if not output_dir.exists():
        return None
    existing = sorted([path for path in output_dir.iterdir() if path.is_dir()], key=lambda p: p.stat().st_mtime)
    return existing[-1] if existing else None


def run_generation(
    prompt: str,
    checkpoint_dir: str,
    tokenizer_dir: str,
    output_dir: str,
    resolution: str,
    sampling_steps: int,
    cfg_scale: float,
    seed: int,
    samples_per_caption: int,
    gpu_ids: str,
    precision: str,
    use_ema: bool,
):
    prompt = prompt.strip()
    if not prompt:
        raise gr.Error("Prompt is required.")

    ckpt_path = Path(checkpoint_dir).expanduser().resolve()
    if not ckpt_path.exists():
        raise gr.Error(f"Checkpoint directory not found: {ckpt_path}")

    tokenizer_path = Path(tokenizer_dir).expanduser().resolve()
    if not tokenizer_path.exists():
        raise gr.Error(f"Tokenizer directory not found: {tokenizer_path}")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    existing_runs = {path.name for path in output_path.iterdir() if path.is_dir()}

    resolution_arg = _normalize_resolution(resolution)
    gpu_ids = gpu_ids.strip()
    if not gpu_ids:
        raise gr.Error("GPU IDs cannot be empty. Example: 0")

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
        raise gr.Error(f"Inference failed.\n\n{logs}")

    run_dir = _detect_new_run(output_path, existing_runs)
    if run_dir is None:
        raise gr.Error(f"Run finished but no output folder was found.\n\n{logs}")

    images = _find_latest_images(run_dir)
    if not images:
        raise gr.Error(f"Run finished but no images were generated.\n\n{logs}")

    summary = (
        f"Checkpoint: {ckpt_path}\n"
        f"Tokenizer: {tokenizer_path}\n"
        f"Resolution: {resolution}\n"
        f"Sampling steps: {sampling_steps}\n"
        f"CFG scale: {cfg_scale}\n"
        f"Seed: {int(seed)}\n"
        f"Samples per prompt: {samples_per_caption}\n"
        f"EMA: {use_ema}\n"
        f"Visible GPU IDs: {gpu_ids}"
    )
    return images, summary, logs


def build_interface():
    with gr.Blocks(title="RetinaLogos Demo") as demo:
        gr.Markdown(
            """
            # RetinaLogos Demo
            Current phase: text-to-fundus generation only.
            This page calls `sample_from_seeds_multigpus.py` and does not assume any mask-condition branch.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=8,
                    placeholder="Describe the retinal image you want to generate.",
                )
                checkpoint_dir = gr.Textbox(
                    label="Checkpoint Directory",
                    value=str(DEFAULT_CHECKPOINT_DIR.resolve()),
                )
                tokenizer_dir = gr.Textbox(
                    label="Tokenizer Directory",
                    value=str(DEFAULT_TOKENIZER_DIR.resolve()),
                )
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value=str(DEFAULT_OUTPUT_DIR.resolve()),
                )
                with gr.Row():
                    resolution = gr.Textbox(label="Resolution", value="512x512")
                    precision = gr.Dropdown(
                        label="Precision",
                        choices=["bf16", "fp16", "fp32", "tf32"],
                        value="bf16",
                    )
                    use_ema = gr.Checkbox(label="Use EMA Weights", value=False)
                with gr.Row():
                    sampling_steps = gr.Slider(label="Sampling Steps", minimum=20, maximum=250, value=50, step=5)
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=8.0, value=4.0, step=0.5)
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    samples_per_caption = gr.Slider(label="Samples per Prompt", minimum=1, maximum=4, value=1, step=1)
                    gpu_ids = gr.Textbox(label="CUDA_VISIBLE_DEVICES", value="0")
                run_button = gr.Button("Run Inference", variant="primary")
            with gr.Column(scale=3):
                gallery = gr.Gallery(label="Generated Images", columns=2, height=520)
                summary = gr.Textbox(label="Run Summary", lines=10)
                logs = gr.Textbox(label="Command Logs", lines=18)

        run_button.click(
            fn=run_generation,
            inputs=[
                prompt,
                checkpoint_dir,
                tokenizer_dir,
                output_dir,
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
    parser = argparse.ArgumentParser(description="Simple Gradio UI for RetinaLogos text-only inference.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_interface().launch(server_name=args.host, server_port=args.port, share=args.share)
