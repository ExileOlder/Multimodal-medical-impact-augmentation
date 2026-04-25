import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_KEYS = [
    "image",
    "image_path",
    "img_path",
    "cfp",
    "cfp_path",
    "path",
]


def resolve_image_path(record: dict, image_root: Path) -> Path | None:
    for key in IMAGE_KEYS:
        value = record.get(key)
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        rooted = (image_root / value).resolve()
        if rooted.exists():
            return rooted
    return None


def load_records(metadata_path: Path) -> list[dict]:
    records: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a real-image subset for FID/KID evaluation.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.jsonl")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory of real images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store sampled real images")
    parser.add_argument("--num_images", type=int, required=True, help="Number of real images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    records = load_records(metadata_path)
    resolved: list[tuple[Path, int]] = []
    for idx, record in enumerate(records):
        image_path = resolve_image_path(record, image_root)
        if image_path is not None:
            resolved.append((image_path, idx))

    if len(resolved) < args.num_images:
        raise ValueError(
            f"Requested {args.num_images} images, but only {len(resolved)} valid image paths were found."
        )

    random.seed(args.seed)
    sampled = random.sample(resolved, args.num_images)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for order, (source_path, record_idx) in enumerate(sampled):
            suffix = source_path.suffix.lower() or ".png"
            target_name = f"real_{order:03d}_{record_idx:06d}{suffix}"
            target_path = output_dir / target_name
            shutil.copy2(source_path, target_path)
            manifest.write(
                json.dumps(
                    {
                        "source_path": str(source_path),
                        "copied_path": str(target_path),
                        "record_index": record_idx,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved {len(sampled)} real images to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
