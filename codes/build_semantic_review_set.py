from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.data_reader import read_general2
from struct_mask_utils import FUSION_MASK_CHANNELS


CAPTION_FIELDS = ["MICROANEURYSMS", "HEMORRHAGES", "EXUDATES", "SOFT_EXUDATES"]
NEGATIVE_HINTS = ("absent", "no ", "not visible", "not present")
POSITIVE_HINTS = ("present", "visible", "multiple", "several", "single", "one ")
LESION_CLASS_NAMES = {"hemorrhages", "soft_exudates", "exudates", "microaneurysms"}
LESION_COLORS = [(name, color) for name, color in FUSION_MASK_CHANNELS if name in LESION_CLASS_NAMES]


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


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")


def lesion_pixel_counts(mask_path: Path) -> dict[str, int]:
    arr = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    counts = {}
    for name, color in LESION_COLORS:
        counts[name] = int(np.all(arr == np.asarray(color, dtype=np.uint8), axis=-1).sum())
    return counts


def dominant_color_name(counts: dict[str, int]) -> str:
    nonzero = {name: value for name, value in counts.items() if value > 0}
    if not nonzero:
        return "none"
    return max(nonzero, key=nonzero.get)


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def render_contact_sheet(entries: list[dict], out_path: Path, root_dir: Path) -> None:
    thumb_w, thumb_h = 240, 240
    padding = 16
    text_h = 42
    row_h = thumb_h + text_h + padding
    width = padding * 3 + thumb_w * 2
    height = padding + row_h * len(entries)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, entry in enumerate(entries):
        top = padding + idx * row_h
        image_path = root_dir / entry["image_rel"]
        mask_path = root_dir / entry["mask_rel"]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        mask.thumbnail((thumb_w, thumb_h))
        img_x = padding
        mask_x = padding * 2 + thumb_w
        img_y = top + text_h
        mask_y = top + text_h
        canvas.paste(image, (img_x, img_y))
        canvas.paste(mask, (mask_x, mask_y))
        title = f"{idx + 1:02d}. {entry['id']} | dom={entry['dominant_color']}"
        subtitle = entry["caption_field_text"][:78]
        draw.text((padding, top), title, fill=(240, 240, 240), font=font)
        draw.text((padding, top + 16), subtitle, fill=(180, 180, 180), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def pick_diverse_examples(candidates: list[dict], limit: int) -> list[dict]:
    by_color: dict[str, list[dict]] = defaultdict(list)
    for item in sorted(candidates, key=lambda x: x["total_lesion_pixels"], reverse=True):
        by_color[item["dominant_color"]].append(item)

    chosen: list[dict] = []
    used_ids = set()

    # First pass: try to cover as many dominant lesion colors as possible.
    for color_name in sorted(by_color.keys()):
        for item in by_color[color_name]:
            if item["id"] not in used_ids:
                chosen.append(item)
                used_ids.add(item["id"])
                break
        if len(chosen) >= limit:
            return chosen[:limit]

    # Second pass: fill with the strongest remaining examples.
    for item in sorted(candidates, key=lambda x: x["total_lesion_pixels"], reverse=True):
        if item["id"] in used_ids:
            continue
        chosen.append(item)
        used_ids.add(item["id"])
        if len(chosen) >= limit:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manual review set of original image + fusion mask + caption.")
    parser.add_argument("--metadata", type=str, default="./data/merged/diabetic/autodl/metadata_train_full_quality_clean.jsonl")
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./results/diagnostics/semantic_review_20")
    parser.add_argument("--per_field", type=int, default=5)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = (code_root / args.metadata).resolve()
    root_dir = Path(args.root_dir)
    if not root_dir.is_absolute():
        root_dir = (code_root / args.root_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (code_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(metadata_path)
    buckets: dict[str, list[dict]] = {field: [] for field in CAPTION_FIELDS}

    for record in records:
        caption = record.get("caption", "")
        statuses = {field: is_present(parse_caption_field(caption, field)) for field in CAPTION_FIELDS}
        active_fields = [field for field, present in statuses.items() if present]
        if len(active_fields) != 1:
            continue
        active_field = active_fields[0]

        image_abs = Path(read_general2(record["image"], str(root_dir)))
        mask_abs = Path(read_general2(record["mask"], str(root_dir)))
        lesion_counts = lesion_pixel_counts(mask_abs)
        total_lesion_pixels = sum(lesion_counts.values())
        if total_lesion_pixels <= 0:
            continue

        buckets[active_field].append(
            {
                "id": record["id"],
                "field": active_field,
                "caption": caption,
                "caption_field_text": parse_caption_field(caption, active_field),
                "image_abs": image_abs,
                "mask_abs": mask_abs,
                "image_rel": str(Path(record["image"])),
                "mask_rel": str(Path(record["mask"])),
                "lesion_pixel_counts": lesion_counts,
                "total_lesion_pixels": total_lesion_pixels,
                "dominant_color": dominant_color_name(lesion_counts),
            }
        )

    selections: dict[str, list[dict]] = {}
    flat_manifest: list[dict] = []

    for field in CAPTION_FIELDS:
        picked = pick_diverse_examples(buckets[field], args.per_field)
        selections[field] = picked
        field_dir = out_dir / safe_name(field.lower())
        field_dir.mkdir(parents=True, exist_ok=True)

        for rank, item in enumerate(picked, start=1):
            sample_dir = field_dir / f"{rank:02d}_{safe_name(item['id'])}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            symlink_force(item["image_abs"], sample_dir / item["image_abs"].name)
            symlink_force(item["mask_abs"], sample_dir / item["mask_abs"].name)
            (sample_dir / "caption.txt").write_text(item["caption"], encoding="utf-8")
            summary = {
                "id": item["id"],
                "field": item["field"],
                "caption_field_text": item["caption_field_text"],
                "dominant_color": item["dominant_color"],
                "lesion_pixel_counts": item["lesion_pixel_counts"],
                "image_path": str(item["image_abs"]),
                "mask_path": str(item["mask_abs"]),
            }
            (sample_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            flat_manifest.append(
                {
                    "field": item["field"],
                    "rank": rank,
                    "id": item["id"],
                    "dominant_color": item["dominant_color"],
                    "lesion_pixel_counts": item["lesion_pixel_counts"],
                    "image": str(sample_dir / item["image_abs"].name),
                    "mask": str(sample_dir / item["mask_abs"].name),
                    "caption_txt": str(sample_dir / "caption.txt"),
                    "summary_json": str(sample_dir / "summary.json"),
                    "caption_field_text": item["caption_field_text"],
                    "caption": item["caption"],
                }
            )

        render_contact_sheet(picked, out_dir / f"{safe_name(field.lower())}_contact_sheet.png", root_dir)

    manifest_json = out_dir / "manifest.json"
    manifest_md = out_dir / "MANIFEST.md"
    manifest_json.write_text(json.dumps(flat_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# Semantic Review Set", ""]
    for field in CAPTION_FIELDS:
        md_lines.append(f"## {field}")
        md_lines.append("")
        md_lines.append(f"Contact sheet: `{(out_dir / f'{safe_name(field.lower())}_contact_sheet.png').name}`")
        md_lines.append("")
        for item in [entry for entry in flat_manifest if entry["field"] == field]:
            md_lines.append(f"### {item['rank']:02d}. {item['id']}")
            md_lines.append(f"- dominant_color: `{item['dominant_color']}`")
            md_lines.append(f"- lesion_pixel_counts: `{json.dumps(item['lesion_pixel_counts'], ensure_ascii=False)}`")
            md_lines.append(f"- image: `{item['image']}`")
            md_lines.append(f"- mask: `{item['mask']}`")
            md_lines.append(f"- caption_txt: `{item['caption_txt']}`")
            md_lines.append(f"- caption_field_text: `{item['caption_field_text']}`")
            md_lines.append("")
        md_lines.append("")
    manifest_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(out_dir)


if __name__ == "__main__":
    main()
