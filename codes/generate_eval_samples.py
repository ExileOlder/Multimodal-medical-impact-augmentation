import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from struct_mask_utils import augment_caption_with_struct_hints


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root_dir / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval samples from metadata records.")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, required=True)
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--num_sampling_steps", type=int, default=80)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--struct_mask_channels", type=int, default=1)
    parser.add_argument("--mask_scale", type=float, default=2.0)
    parser.add_argument("--vae", type=str, default="sdxl")
    parser.add_argument(
        "--fundus_color_reference_mode",
        type=str,
        choices=["none", "global", "source_image"],
        default="none",
        help="Use no color reference, one shared global reference, or each record's source image.",
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

    script_path = Path(__file__).resolve().parent / "inference_mask.py"
    metadata_path = Path(args.metadata).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(metadata_path)[: args.limit]
    if not records:
        raise RuntimeError(f"No records found in {metadata_path}")

    color_reference_mode = args.fundus_color_reference_mode
    if color_reference_mode == "none" and args.fundus_color_reference:
        color_reference_mode = "global"
    if color_reference_mode == "global" and not args.fundus_color_reference:
        raise ValueError("--fundus_color_reference is required when --fundus_color_reference_mode=global")

    for idx, record in enumerate(records, start=1):
        record_id = str(record["id"])
        mask_path = resolve_path(root_dir, record["mask"])
        image_value = record.get("image") or record.get("image_path")
        prompt = augment_caption_with_struct_hints(
            str(record["caption"]),
            image_path=image_value,
            mask_path=mask_path,
            struct_mask_channels=args.struct_mask_channels,
        )
        cmd = [
            sys.executable,
            str(script_path),
            "--base_ckpt",
            args.base_ckpt,
            "--adapter_ckpt",
            args.adapter_ckpt,
            "--prompt",
            prompt,
            "--mask_path",
            str(mask_path),
            "--out_dir",
            str(out_dir),
            "--output_name",
            record_id,
            "--seed",
            str(args.seed),
            "--cfg_scale",
            str(args.cfg_scale),
            "--num_sampling_steps",
            str(args.num_sampling_steps),
            "--precision",
            args.precision,
            "--vae",
            args.vae,
            "--image_size",
            str(args.image_size),
            "--struct_mask_channels",
            str(args.struct_mask_channels),
            "--mask_scale",
            str(args.mask_scale),
            "--qk_norm",
        ]
        color_reference_path = None
        if color_reference_mode == "global":
            color_reference_path = args.fundus_color_reference
        elif color_reference_mode == "source_image":
            if image_value is None:
                raise ValueError(
                    f"Record {record_id} is missing image/image_path, cannot use source-image color reference."
                )
            color_reference_path = str(resolve_path(root_dir, image_value))

        if color_reference_path:
            cmd.extend(
                [
                    "--fundus_color_reference",
                    color_reference_path,
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
            )
            if args.fundus_profile_json:
                cmd.extend(["--fundus_profile_json", args.fundus_profile_json])
        print(f"[{idx}/{len(records)}] generate {record_id}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
