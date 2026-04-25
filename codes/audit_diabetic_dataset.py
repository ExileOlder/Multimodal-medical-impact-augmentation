import argparse
import json
import random
import textwrap
from collections import Counter
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Pillow is required for audit_diabetic_dataset.py. Install it in your training environment first."
    ) from exc


def load_metadata(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return records


def resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def safe_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def draw_triplet_panel(image: Image.Image, mask: Image.Image, caption: str, output_path: Path) -> None:
    image = image.convert("RGB").resize((512, 512))
    mask = mask.convert("L").resize((512, 512))
    mask_rgb = Image.merge("RGB", (mask, Image.new("L", mask.size, 0), Image.new("L", mask.size, 0)))
    overlay = Image.blend(image, mask_rgb, 0.35)

    caption_lines = textwrap.wrap(caption, width=62)[:18]
    caption_block = "\n".join(caption_lines)

    canvas = Image.new("RGB", (1536, 760), (255, 255, 255))
    canvas.paste(image, (0, 0))
    canvas.paste(mask.convert("RGB"), (512, 0))
    canvas.paste(overlay, (1024, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((20, 520), "image", fill=(0, 0, 0))
    draw.text((532, 520), "mask", fill=(0, 0, 0))
    draw.text((1044, 520), "overlay", fill=(0, 0, 0))
    draw.text((20, 560), caption_block, fill=(0, 0, 0), spacing=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rebuilt diabetic metadata and export quick visualizations.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata jsonl")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory used by metadata relative paths")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for audit report and visuals")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of random visualization samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_records",
        type=int,
        default=0,
        help="Limit processed records for lightweight audit. 0 means full dataset.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    records = load_metadata(metadata_path)
    if args.max_records > 0:
        records = records[: args.max_records]
    stats = Counter()
    caption_lengths: list[int] = []
    image_sizes = Counter()
    mismatched_size_examples: list[dict] = []
    unreadable_examples: list[dict] = []

    for record in records:
        stats["total_records"] += 1
        image_path = resolve_path(root_dir, record["image"])
        mask_path = resolve_path(root_dir, record["mask"])
        caption = record.get("caption", "")
        caption_lengths.append(len(caption))

        if not image_path.exists():
            stats["missing_image"] += 1
            continue
        if not mask_path.exists():
            stats["missing_mask"] += 1
            continue

        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image.load()
                mask.load()
                image_size = image.size
                mask_size = mask.size
                image_sizes[str(image_size)] += 1
                if image_size != mask_size:
                    stats["size_mismatch"] += 1
                    if len(mismatched_size_examples) < 20:
                        mismatched_size_examples.append(
                            {
                                "id": record.get("id"),
                                "image_size": image_size,
                                "mask_size": mask_size,
                            }
                        )

                mask_gray = mask.convert("L")
                min_v, max_v = mask_gray.getextrema()
                if max_v == 0:
                    stats["empty_mask"] += 1
                if not caption.strip():
                    stats["empty_caption"] += 1
                if len(caption) > 1024:
                    stats["long_caption_gt_1024"] += 1
                if len(caption) > 1536:
                    stats["long_caption_gt_1536"] += 1
                if min_v == max_v:
                    stats["flat_mask"] += 1
        except (OSError, ValueError) as exc:
            stats["unreadable_file"] += 1
            if len(unreadable_examples) < 20:
                unreadable_examples.append(
                    {
                        "id": record.get("id"),
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sampled_records = rng.sample(records, k=min(args.num_samples, len(records)))
    visual_dir = output_dir / "samples"
    for order, record in enumerate(sampled_records):
        image_path = resolve_path(root_dir, record["image"])
        mask_path = resolve_path(root_dir, record["mask"])
        if not image_path.exists() or not mask_path.exists():
            continue
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                draw_triplet_panel(
                    image=image,
                    mask=mask,
                    caption=record.get("caption", ""),
                    output_path=visual_dir / f"sample_{order:02d}_{record.get('id', 'unknown')}.png",
                )
        except (OSError, ValueError):
            continue

    caption_lengths_sorted = sorted(caption_lengths)
    report = {
        "metadata": str(metadata_path),
        "root_dir": str(root_dir),
        "total_records": stats["total_records"],
        "missing_image": stats["missing_image"],
        "missing_mask": stats["missing_mask"],
        "size_mismatch": stats["size_mismatch"],
        "unreadable_file": stats["unreadable_file"],
        "empty_mask": stats["empty_mask"],
        "empty_caption": stats["empty_caption"],
        "long_caption_gt_1024": stats["long_caption_gt_1024"],
        "long_caption_gt_1536": stats["long_caption_gt_1536"],
        "flat_mask": stats["flat_mask"],
        "missing_image_pct": safe_percent(stats["missing_image"], stats["total_records"]),
        "missing_mask_pct": safe_percent(stats["missing_mask"], stats["total_records"]),
        "size_mismatch_pct": safe_percent(stats["size_mismatch"], stats["total_records"]),
        "unreadable_file_pct": safe_percent(stats["unreadable_file"], stats["total_records"]),
        "empty_mask_pct": safe_percent(stats["empty_mask"], stats["total_records"]),
        "caption_length": {
            "min": caption_lengths_sorted[0] if caption_lengths_sorted else 0,
            "median": caption_lengths_sorted[len(caption_lengths_sorted) // 2] if caption_lengths_sorted else 0,
            "max": caption_lengths_sorted[-1] if caption_lengths_sorted else 0,
            "mean": round(sum(caption_lengths) / len(caption_lengths), 4) if caption_lengths else 0,
        },
        "top_image_sizes": image_sizes.most_common(10),
        "sample_size_mismatches": mismatched_size_examples,
        "sample_unreadable": unreadable_examples,
        "sample_visual_dir": str(visual_dir),
    }
    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total records: {stats['total_records']}")
    print(f"Empty masks: {stats['empty_mask']}")
    print(f"Size mismatches: {stats['size_mismatch']}")
    print(f"Unreadable files: {stats['unreadable_file']}")
    print(f"Audit report: {report_path}")
    print(f"Visual samples: {visual_dir}")


if __name__ == "__main__":
    main()
