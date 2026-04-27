import argparse
import subprocess
import sys
from pathlib import Path


def resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def latest_checkpoint_dir(run_dir: Path) -> Path:
    checkpoint_root = run_dir / "checkpoints"
    ckpt_dirs = sorted(
        p for p in checkpoint_root.glob("*")
        if p.is_dir() and p.name.isdigit()
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No numeric checkpoint directories found under {checkpoint_root}")
    return ckpt_dirs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified structural evaluation entrypoint.")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--step_dir", type=str, default=None)
    parser.add_argument("--base_ckpt", type=str, default="./../checkpoints")
    parser.add_argument("--metadata", type=str, default="./data/merged/diabetic/autodl/metadata_example_20_maskqc_clean_colorstyle8.jsonl")
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--generated_dir", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--adapter_ckpt", type=str, default=None)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--num_sampling_steps", type=int, default=80)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--struct_mask_channels", type=int, default=6)
    parser.add_argument("--mask_scale", type=float, default=1.0)
    parser.add_argument("--vae", type=str, default="sdxl")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--alignment_mode",
        type=str,
        choices=["canonical_square", "raw_resize"],
        default="canonical_square",
    )
    parser.add_argument("--fundus_padding_ratio", type=float, default=0.01)
    parser.add_argument(
        "--fundus_color_reference_mode",
        type=str,
        choices=["none", "global", "source_image"],
        default="none",
    )
    parser.add_argument("--fundus_color_reference", type=str, default=None)
    parser.add_argument("--fundus_color_gamma", type=float, default=1.0)
    parser.add_argument("--fundus_warm_r", type=float, default=1.0)
    parser.add_argument("--fundus_warm_g", type=float, default=1.0)
    parser.add_argument("--fundus_warm_b", type=float, default=1.0)
    parser.add_argument("--fundus_saturation", type=float, default=1.0)
    parser.add_argument("--fundus_contrast", type=float, default=1.0)
    parser.add_argument("--fundus_shadow_lift", type=float, default=0.0)
    parser.add_argument("--fundus_edge_lift", type=float, default=0.0)
    parser.add_argument("--fundus_flatten_strength", type=float, default=0.0)
    parser.add_argument("--fundus_profile_json", type=str, default=None)
    parser.add_argument("--fundus_shade_correction_strength", type=float, default=0.0)
    parser.add_argument("--fundus_shade_blur_radius", type=float, default=41.0)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    metadata = resolve(args.metadata if Path(args.metadata).is_absolute() else str(code_root / args.metadata))
    root_dir = resolve(args.root_dir if Path(args.root_dir).is_absolute() else str(code_root / args.root_dir))
    base_ckpt = resolve(args.base_ckpt if Path(args.base_ckpt).is_absolute() else str(code_root / args.base_ckpt))

    run_dir = resolve(args.run_dir) if args.run_dir else None
    step_dir = resolve(args.step_dir) if args.step_dir else None
    if run_dir and not step_dir:
        step_dir = latest_checkpoint_dir(run_dir)
    if step_dir and not args.adapter_ckpt:
        args.adapter_ckpt = str(step_dir / "adapter.pth")
    adapter_ckpt = resolve(args.adapter_ckpt) if args.adapter_ckpt else None

    if args.generated_dir:
        generated_dir = resolve(args.generated_dir)
    elif step_dir:
        generated_dir = step_dir.parent / "eval" / step_dir.name / "generated"
    else:
        raise ValueError("Either --generated_dir or --run_dir/--step_dir must be provided.")

    if args.output_json:
        output_json = resolve(args.output_json)
    elif step_dir:
        output_json = step_dir.parent / "eval" / step_dir.name / "structural_metrics.json"
    else:
        output_json = generated_dir / "structural_metrics.json"

    if not metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata}")
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint dir not found: {base_ckpt}")

    generated_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_generation:
        if adapter_ckpt is None or not adapter_ckpt.exists():
            raise FileNotFoundError("Adapter checkpoint is required for generation.")
        gen_cmd = [
            sys.executable,
            str(code_root / "generate_eval_samples.py"),
            "--metadata",
            str(metadata),
            "--root_dir",
            str(root_dir),
            "--base_ckpt",
            str(base_ckpt),
            "--adapter_ckpt",
            str(adapter_ckpt),
            "--out_dir",
            str(generated_dir),
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
            "--cfg_scale",
            str(args.cfg_scale),
            "--num_sampling_steps",
            str(args.num_sampling_steps),
            "--precision",
            args.precision,
            "--image_size",
            str(args.image_size),
            "--struct_mask_channels",
            str(args.struct_mask_channels),
            "--mask_scale",
            str(args.mask_scale),
            "--vae",
            args.vae,
            "--fundus_color_reference_mode",
            args.fundus_color_reference_mode,
            "--fundus_color_gamma",
            str(args.fundus_color_gamma),
            "--fundus_warm_r",
            str(args.fundus_warm_r),
            "--fundus_warm_g",
            str(args.fundus_warm_g),
            "--fundus_warm_b",
            str(args.fundus_warm_b),
            "--fundus_saturation",
            str(args.fundus_saturation),
            "--fundus_contrast",
            str(args.fundus_contrast),
            "--fundus_shadow_lift",
            str(args.fundus_shadow_lift),
            "--fundus_edge_lift",
            str(args.fundus_edge_lift),
            "--fundus_flatten_strength",
            str(args.fundus_flatten_strength),
            "--fundus_shade_correction_strength",
            str(args.fundus_shade_correction_strength),
            "--fundus_shade_blur_radius",
            str(args.fundus_shade_blur_radius),
        ]
        if args.fundus_color_reference:
            gen_cmd.extend(["--fundus_color_reference", str(resolve(args.fundus_color_reference))])
        if args.fundus_profile_json:
            gen_cmd.extend(["--fundus_profile_json", str(resolve(args.fundus_profile_json))])
        subprocess.run(gen_cmd, check=True)

    eval_cmd = [
        sys.executable,
        str(code_root / "eval_structural_metrics.py"),
        "--metadata",
        str(metadata),
        "--generated_dir",
        str(generated_dir),
        "--root_dir",
        str(root_dir),
        "--output_json",
        str(output_json),
        "--topk",
        str(args.topk),
        "--alignment_mode",
        args.alignment_mode,
        "--fundus_padding_ratio",
        str(args.fundus_padding_ratio),
    ]
    subprocess.run(eval_cmd, check=True)
    print(f"[ok] structural eval saved to {output_json}")


if __name__ == "__main__":
    main()
