import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def load_caption_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            records.append(record)
    return records


def index_files_by_stem(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        index[path.stem] = path
    return index


def choose_image_path(record: dict, image_index: dict[str, Path]) -> Path | None:
    record_id = str(record["id"])
    old_path = record.get("fundus_path")
    candidates = [record_id]
    if old_path:
        candidates.append(Path(old_path).stem)
    for key in candidates:
        path = image_index.get(key)
        if path is not None:
            return path
    return None


def choose_mask_path(record: dict, mask_index: dict[str, Path]) -> Path | None:
    record_id = str(record["id"])
    old_path = record.get("mask_path")
    candidates = [f"{record_id}_fusion", record_id]
    if old_path:
        candidates.append(Path(old_path).stem)
    for key in candidates:
        path = mask_index.get(key)
        if path is not None:
            return path
    return None


def build_merged_image_index(image_dirs: list[Path]) -> dict[str, Path]:
    merged: dict[str, Path] = {}
    for image_dir in image_dirs:
        for stem, path in index_files_by_stem(image_dir).items():
            if stem not in merged:
                merged[stem] = path
    return merged


def make_output_record(
    record: dict,
    image_path: Path,
    mask_path: Path,
    root_dir: Path,
    keep_raw_paths: bool,
    image_size: tuple[int, int] | None = None,
    mask_size: tuple[int, int] | None = None,
) -> dict:
    output = {
        "id": str(record["id"]),
        "image": str(image_path.relative_to(root_dir)).replace("\\", "/"),
        "mask": str(mask_path.relative_to(root_dir)).replace("\\", "/"),
        "caption": record.get("caption", ""),
        "caption_source": "fundus_captions.jsonl.json",
        "mask_name": mask_path.name,
    }
    if image_size is not None:
        output["image_size"] = list(image_size)
    if mask_size is not None:
        output["mask_size"] = list(mask_size)
    if keep_raw_paths:
        output["raw_fundus_path"] = record.get("fundus_path")
        output["raw_mask_path"] = record.get("mask_path")
        output["valid_format"] = record.get("valid_format")
    return output


def make_excluded_record(
    record: dict,
    reasons: list[str],
    image_path: Path | None,
    mask_path: Path | None,
    keep_raw_paths: bool,
    image_size: tuple[int, int] | None = None,
    mask_size: tuple[int, int] | None = None,
) -> dict:
    output = {
        "id": str(record.get("id")),
        "exclude_reason": "+".join(reasons),
        "exclude_reasons": reasons,
        "image_found": image_path is not None,
        "mask_found": mask_path is not None,
        "caption_present": bool(record.get("caption", "").strip()),
    }
    if image_path is not None:
        output["resolved_image_path"] = str(image_path)
    if mask_path is not None:
        output["resolved_mask_path"] = str(mask_path)
    if image_size is not None:
        output["image_size"] = list(image_size)
    if mask_size is not None:
        output["mask_size"] = list(mask_size)
    if keep_raw_paths:
        output["raw_fundus_path"] = record.get("fundus_path")
        output["raw_mask_path"] = record.get("mask_path")
        output["valid_format"] = record.get("valid_format")
    return output


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def inspect_image_file(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build data-ready metadata from diabetic fundus captions, real images, and masks."
    )
    parser.add_argument("--captions", type=str, required=True, help="Path to fundus_captions.jsonl.json")
    parser.add_argument("--image_dir", type=str, required=True, help="Primary directory containing Kaggle fundus images")
    parser.add_argument(
        "--extra_image_dir",
        type=str,
        action="append",
        default=[],
        help="Additional image directories to merge into the image index. Can be passed multiple times.",
    )
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store rebuilt metadata")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory used to write relative paths. Defaults to output_dir parent if omitted.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument(
        "--example_count",
        type=int,
        default=20,
        help="Number of records held out from metadata_all for lightweight example tests (not used for training).",
    )
    parser.add_argument(
        "--allow_size_mismatch",
        action="store_true",
        help="Keep matched records even when real image size and mask size differ; the mismatch is still recorded.",
    )
    parser.add_argument(
        "--keep_raw_paths",
        action="store_true",
        help="Keep old caption file path fields in output for debugging.",
    )
    args = parser.parse_args()

    captions_path = Path(args.captions).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    extra_image_dirs = [Path(p).expanduser().resolve() for p in args.extra_image_dir]
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve() if args.root_dir else output_dir.parent.resolve()

    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    for extra_dir in extra_image_dirs:
        if not extra_dir.exists():
            raise FileNotFoundError(f"Extra image directory not found: {extra_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    if root_dir not in [output_dir, *output_dir.parents]:
        raise ValueError(f"root_dir must be an ancestor of output_dir so relative paths remain valid: {root_dir}")
    if not 0 <= args.val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {args.val_ratio}")
    if args.example_count < 0:
        raise ValueError(f"example_count must be >= 0, got {args.example_count}")

    records = load_caption_records(captions_path)
    image_dirs = [image_dir, *extra_image_dirs]
    image_index = build_merged_image_index(image_dirs)
    mask_index = index_files_by_stem(mask_dir)

    stats = Counter()
    matched: list[dict] = []
    excluded: list[dict] = []

    for record in records:
        stats["caption_records"] += 1
        image_path = choose_image_path(record, image_index)
        mask_path = choose_mask_path(record, mask_index)
        caption = record.get("caption", "")
        reasons: list[str] = []
        image_size: tuple[int, int] | None = None
        mask_size: tuple[int, int] | None = None

        if image_path is None:
            reasons.append("missing_image")
        if mask_path is None:
            reasons.append("missing_mask")
        if not caption.strip():
            reasons.append("empty_caption")

        if image_path is not None and mask_path is not None:
            image_size, image_err = inspect_image_file(image_path)
            mask_size, mask_err = inspect_image_file(mask_path)
            if image_err or mask_err:
                reasons.append("unreadable_file")
            if image_err:
                stats["unreadable_image"] += 1
            if mask_err:
                stats["unreadable_mask"] += 1
            if image_size is not None and mask_size is not None and image_size != mask_size:
                stats["size_mismatch"] += 1
                if not args.allow_size_mismatch:
                    reasons.append("size_mismatch")

        if reasons:
            stats["excluded_records"] += 1
            for reason in reasons:
                stats[reason] += 1
            excluded.append(
                make_excluded_record(
                    record=record,
                    reasons=reasons,
                    image_path=image_path,
                    mask_path=mask_path,
                    keep_raw_paths=args.keep_raw_paths,
                    image_size=image_size,
                    mask_size=mask_size,
                )
            )
            continue

        matched.append(
            make_output_record(
                record,
                image_path,
                mask_path,
                root_dir,
                args.keep_raw_paths,
                image_size=image_size,
                mask_size=mask_size,
            )
        )
        stats["matched_records"] += 1

    if not matched:
        raise RuntimeError("No matched records were found. Check image_dir, mask_dir, and naming rules.")

    matched = sorted(matched, key=lambda item: item["id"])
    shuffled = matched[:]
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)

    val_size = int(len(shuffled) * args.val_ratio)
    if args.val_ratio > 0 and val_size == 0 and len(shuffled) > 1:
        val_size = 1
    val_records = sorted(shuffled[:val_size], key=lambda item: item["id"])
    train_records = sorted(shuffled[val_size:], key=lambda item: item["id"])
    example_size = min(args.example_count, len(train_records))
    example_records = sorted(train_records[:example_size], key=lambda item: item["id"])
    train_full_records = sorted(train_records[example_size:], key=lambda item: item["id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_all = output_dir / "metadata_all.jsonl"
    metadata_train = output_dir / "metadata_train.jsonl"
    metadata_val = output_dir / "metadata_val.jsonl"
    metadata_train_full = output_dir / "metadata_train_full.jsonl"
    metadata_example = output_dir / "metadata_example_20.jsonl"
    excluded_path = output_dir / "excluded_records.jsonl"
    report_path = output_dir / "build_report.json"

    save_jsonl(metadata_all, matched)
    save_jsonl(metadata_train, train_records)
    save_jsonl(metadata_val, val_records)
    save_jsonl(metadata_train_full, train_full_records)
    save_jsonl(metadata_example, example_records)
    save_jsonl(excluded_path, excluded)

    report = {
        "captions_path": str(captions_path),
        "image_dir": str(image_dir),
        "image_dirs": [str(p) for p in image_dirs],
        "mask_dir": str(mask_dir),
        "root_dir": str(root_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "example_count": args.example_count,
        "caption_records": len(records),
        "source_image_files": len(image_index),
        "mask_files": len(mask_index),
        "matched_records": len(matched),
        "excluded_records": len(excluded),
        "train_records": len(train_records),
        "train_full_records": len(train_full_records),
        "example_records": len(example_records),
        "val_records": len(val_records),
        "stats": dict(stats),
        "sample_excluded": excluded[:100],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Caption records: {len(records)}")
    print(f"Source image files: {len(image_index)}")
    print(f"Mask files: {len(mask_index)}")
    print(f"Matched records: {len(matched)}")
    print(f"Excluded records: {len(excluded)}")
    print(f"Train records: {len(train_records)}")
    print(f"Train full records: {len(train_full_records)}")
    print(f"Example records: {len(example_records)}")
    print(f"Val records: {len(val_records)}")
    print(f"Excluded file: {excluded_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
