import argparse
import json
import os
from pathlib import Path

from PIL import Image
from torchvision import transforms
import yaml

from struct_mask_utils import get_struct_mask_channel_names, load_struct_mask_tensor


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_text_contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8", errors="ignore")


def build_parser():
    parser = argparse.ArgumentParser(description="CPU-only preflight validation for diabetic mask training.")
    parser.add_argument("--data_config", default="./configs/train/diabetic_merged_mask.yaml")
    parser.add_argument("--code_root", default=None, help="Defaults to the directory containing this script.")
    parser.add_argument("--base_ckpt", default="../checkpoints/consolidated.00-of-01.pth")
    parser.add_argument("--tokenizer_dir", default="./google_gemma-2b")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--struct_mask_channels", type=int, default=1)
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument("--output_json", default="./results/preflight/training_ready_report.json")
    return parser


def main():
    args = build_parser().parse_args()
    code_root = Path(args.code_root).resolve() if args.code_root else Path(__file__).resolve().parent

    data_config_path = resolve_path(code_root, args.data_config)
    base_ckpt_path = resolve_path(code_root, args.base_ckpt)
    tokenizer_dir = resolve_path(code_root, args.tokenizer_dir)
    output_json = resolve_path(code_root, args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "ok",
        "code_root": str(code_root),
        "data_config": str(data_config_path),
        "base_ckpt": str(base_ckpt_path),
        "tokenizer_dir": str(tokenizer_dir),
        "checks": {},
        "sample_stats": {},
        "fatal_errors": [],
        "warnings": [],
    }
    report["checks"]["struct_mask_channels"] = args.struct_mask_channels
    report["checks"]["struct_mask_channel_names"] = get_struct_mask_channel_names(args.struct_mask_channels)

    if not data_config_path.exists():
        report["fatal_errors"].append(f"Missing data config: {data_config_path}")
    if not base_ckpt_path.exists():
        report["fatal_errors"].append(f"Missing base checkpoint: {base_ckpt_path}")
    if not tokenizer_dir.exists():
        report["fatal_errors"].append(f"Missing tokenizer directory: {tokenizer_dir}")

    metadata_path = None
    image_root = None
    if data_config_path.exists():
        cfg = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
        meta_items = cfg.get("META") or []
        if not meta_items:
            report["fatal_errors"].append("META section is empty in data config.")
        else:
            first = meta_items[0]
            metadata_path = resolve_path(code_root, first["path"])
            image_root = resolve_path(code_root, first["image_root"])
            report["checks"]["metadata_path"] = str(metadata_path)
            report["checks"]["image_root"] = str(image_root)
            if not metadata_path.exists():
                report["fatal_errors"].append(f"Missing metadata file: {metadata_path}")
            if not image_root.exists():
                report["fatal_errors"].append(f"Missing image root: {image_root}")

    train_py = code_root / "train.py"
    smoke_sh = code_root / "configs/train/run_4090_smoke_setup.sh"
    report["checks"]["train_py_has_old_abs_path"] = check_text_contains(train_py, "/home/maziheng")
    report["checks"]["smoke_uses_unified_entry"] = check_text_contains(smoke_sh, "run_fullmask_train.sh")
    if report["checks"]["train_py_has_old_abs_path"]:
        report["fatal_errors"].append("train.py still contains /home/maziheng absolute path.")
    if not report["checks"]["smoke_uses_unified_entry"]:
        report["warnings"].append("Smoke setup script does not invoke run_fullmask_train.sh.")

    if metadata_path and metadata_path.exists() and image_root and image_root.exists():
        records = list(read_jsonl(metadata_path))
        report["sample_stats"]["metadata_records"] = len(records)
        if not records:
            report["fatal_errors"].append("Metadata file is empty.")
        else:
            image_transform = transforms.Compose(
                [
                    transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                ]
            )
            sample_records = records[: min(args.sample_count, len(records))]
            readable_images = 0
            readable_masks = 0
            mismatched_original_sizes = 0
            empty_masks = 0
            transformed_image_shapes = set()
            transformed_mask_shapes = set()

            for rec in sample_records:
                image_path = image_root / rec["image"]
                mask_path = image_root / rec["mask"]
                if not image_path.exists():
                    report["fatal_errors"].append(f"Missing image in sample set: {image_path}")
                    continue
                if not mask_path.exists():
                    report["fatal_errors"].append(f"Missing mask in sample set: {mask_path}")
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")
                    readable_images += 1
                except Exception as exc:
                    report["fatal_errors"].append(f"Unreadable image: {image_path} ({type(exc).__name__})")
                    continue

                try:
                    mask_tensor = load_struct_mask_tensor(
                        mask_path,
                        image_size=args.image_size,
                        struct_mask_channels=args.struct_mask_channels,
                    )
                    readable_masks += 1
                except Exception as exc:
                    report["fatal_errors"].append(f"Unreadable mask: {mask_path} ({type(exc).__name__})")
                    continue

                mask = Image.open(mask_path)
                if image.size != mask.size:
                    mismatched_original_sizes += 1

                image_tensor = image_transform(image)
                transformed_image_shapes.add(tuple(image_tensor.shape))
                transformed_mask_shapes.add(tuple(mask_tensor.shape))
                if float(mask_tensor.max()) <= 0.0:
                    empty_masks += 1

            report["sample_stats"]["sample_checked"] = len(sample_records)
            report["sample_stats"]["readable_images"] = readable_images
            report["sample_stats"]["readable_masks"] = readable_masks
            report["sample_stats"]["mismatched_original_sizes_in_sample"] = mismatched_original_sizes
            report["sample_stats"]["empty_masks_in_sample"] = empty_masks
            report["sample_stats"]["transformed_image_shapes"] = sorted(list(transformed_image_shapes))
            report["sample_stats"]["transformed_mask_shapes"] = sorted(list(transformed_mask_shapes))

            if transformed_image_shapes != {(3, args.image_size, args.image_size)}:
                report["fatal_errors"].append("Unexpected transformed image shape found in sample check.")
            if transformed_mask_shapes != {(args.struct_mask_channels, args.image_size, args.image_size)}:
                report["fatal_errors"].append("Unexpected transformed mask shape found in sample check.")

    report["status"] = "failed" if report["fatal_errors"] else "ok"
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(1 if report["fatal_errors"] else 0)


if __name__ == "__main__":
    main()
