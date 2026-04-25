from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from data.data_reader import read_general2
from struct_mask_utils import FUSION_MASK_CHANNELS


LESION_COLOR_NAMES = ["hemorrhages", "soft_exudates", "exudates", "microaneurysms"]
CAPTION_FIELDS = ["MICROANEURYSMS", "HEMORRHAGES", "EXUDATES", "SOFT_EXUDATES"]
NEGATIVE_HINTS = ("absent", "no ", "not visible", "not present")
POSITIVE_HINTS = ("present", "visible", "multiple", "several", "single", "one ")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_caption_field(caption: str, field: str) -> str:
    match = re.search(rf"{field}:\s*([^\.]+)", caption, flags=re.I)
    return match.group(1).strip().lower() if match else "missing"


def is_present(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in NEGATIVE_HINTS):
        return False
    return any(token in lowered for token in POSITIVE_HINTS)


def lesion_pixel_counts(mask_path: Path) -> dict[str, int]:
    arr = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    counts = {}
    for name, color in FUSION_MASK_CHANNELS:
        if name not in LESION_COLOR_NAMES:
            continue
        counts[name] = int(np.all(arr == np.asarray(color, dtype=np.uint8), axis=-1).sum())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the semantic alignment between fusion-mask lesion colors and caption disease fields.")
    parser.add_argument("--metadata", type=str, default="./data/merged/diabetic/autodl/metadata_train_full_quality_clean.jsonl")
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./results/diagnostics/fusion_mask_semantics_audit.json")
    parser.add_argument("--example_limit", type=int, default=8)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = (code_root / args.metadata).resolve()
    root_dir = Path(args.root_dir)
    if not root_dir.is_absolute():
        root_dir = (code_root / args.root_dir).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (code_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(metadata_path)

    color_presence_by_field = {name: Counter() for name in LESION_COLOR_NAMES}
    dominant_color_by_field = {name: Counter() for name in LESION_COLOR_NAMES}
    single_field_dominant = {field: Counter() for field in CAPTION_FIELDS}
    single_field_examples: dict[str, list[dict]] = defaultdict(list)
    color_record_presence = Counter()
    dominant_color_records = Counter()

    for record in records:
        caption = record.get("caption", "")
        field_status = {field: is_present(parse_caption_field(caption, field)) for field in CAPTION_FIELDS}
        mask_path = Path(read_general2(record["mask"], str(root_dir)))
        lesion_counts = lesion_pixel_counts(mask_path)
        nonzero_counts = {name: count for name, count in lesion_counts.items() if count > 0}
        if not nonzero_counts:
            continue

        dominant_color = max(nonzero_counts, key=nonzero_counts.get)
        dominant_color_records[dominant_color] += 1

        for color_name in nonzero_counts:
            color_record_presence[color_name] += 1
            for field, present in field_status.items():
                if present:
                    color_presence_by_field[color_name][field] += 1

        for field, present in field_status.items():
            if present:
                dominant_color_by_field[dominant_color][field] += 1

        active_fields = [field for field, present in field_status.items() if present]
        if len(active_fields) == 1:
            only_field = active_fields[0]
            single_field_dominant[only_field][dominant_color] += 1
            if len(single_field_examples[only_field]) < args.example_limit:
                single_field_examples[only_field].append(
                    {
                        "image": record.get("image"),
                        "mask": record.get("mask"),
                        "dominant_color": dominant_color,
                        "lesion_pixel_counts": nonzero_counts,
                        "caption_excerpt": caption[:320],
                    }
                )

    report = {
        "metadata": str(metadata_path),
        "root_dir": str(root_dir),
        "records_total": len(records),
        "lesion_color_names": LESION_COLOR_NAMES,
        "caption_fields": CAPTION_FIELDS,
        "color_record_presence": dict(color_record_presence),
        "dominant_color_records": dict(dominant_color_records),
        "color_presence_by_caption_field": {
            color_name: dict(counter) for color_name, counter in color_presence_by_field.items()
        },
        "dominant_color_by_caption_field": {
            color_name: dict(counter) for color_name, counter in dominant_color_by_field.items()
        },
        "single_field_dominant_color": {
            field: dict(counter) for field, counter in single_field_dominant.items()
        },
        "single_field_examples": single_field_examples,
        "conclusion": (
            "This report is intended to verify whether lesion colors map cleanly to caption disease fields. "
            "If multiple lesion colors dominate within records where only one caption field is positive, then "
            "the color-to-disease mapping is not a simple one-to-one correspondence."
        ),
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
