from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from data.data_reader import read_general2
from struct_mask_utils import FUSION_MASK_CHANNELS


LESION_CLASS_NAMES = ("hemorrhages", "soft_exudates", "exudates", "microaneurysms")
LESION_FIELD_MAP = {
    "hemorrhages": "HEMORRHAGES",
    "soft_exudates": "SOFT_EXUDATES",
    "exudates": "EXUDATES",
    "microaneurysms": "MICROANEURYSMS",
}
LESION_COLORS = {name: color for name, color in FUSION_MASK_CHANNELS if name in LESION_CLASS_NAMES}
NEGATIVE_HINTS = ("absent", "no ", "not visible", "not present", "none")
POSITIVE_HINTS = ("present", "visible", "multiple", "several", "single", "one ", "few", "scattered")


def lesion_name_to_text(lesion_name: str) -> str:
    return lesion_name.replace("_", " ")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_caption_field(caption: str, field: str) -> str:
    match = re.search(rf"{field}:\s*([^\.]+)", caption or "", flags=re.I)
    if not match:
        return ""
    return match.group(1).strip()


def caption_supports_lesion(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if any(token in lowered for token in NEGATIVE_HINTS):
        return False
    if any(token in lowered for token in POSITIVE_HINTS):
        return True
    return True


def build_local_caption(
    lesion_name: str,
    caption_field_text: str,
    *,
    vessel_ratio_in_crop: float,
    optic_disc_present_in_crop: bool,
) -> str:
    lesion_text = lesion_name_to_text(lesion_name)
    description = (caption_field_text or "").strip()
    if not description:
        description = f"The crop should show medically plausible {lesion_text} findings."

    context_bits = [f"LOCAL_TASK: edit the local retinal crop to express {lesion_text}."]
    context_bits.append(f"LESION_DESCRIPTION: {description}")
    if vessel_ratio_in_crop >= 0.01:
        context_bits.append("LOCAL_CONTEXT: retinal vessels are visible in this crop.")
    if optic_disc_present_in_crop:
        context_bits.append("LOCAL_CONTEXT: optic disc tissue is visible in this crop.")
    context_bits.append("LOCAL_CONSTRAINT: preserve realistic retinal color and background texture.")
    return " ".join(context_bits)


def connected_components(binary_mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """
    Return connected components as (y_min, x_min, y_max, x_max, area), where max bounds are inclusive.
    """
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=np.uint8)
    components: list[tuple[int, int, int, int, int]] = []

    ys, xs = np.where(binary_mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = 1
        area = 0
        y_min = y_max = y0
        x_min = x_max = x0
        while stack:
            y, x = stack.pop()
            area += 1
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y > 0 and binary_mask[y - 1, x] and not visited[y - 1, x]:
                visited[y - 1, x] = 1
                stack.append((y - 1, x))
            if y + 1 < height and binary_mask[y + 1, x] and not visited[y + 1, x]:
                visited[y + 1, x] = 1
                stack.append((y + 1, x))
            if x > 0 and binary_mask[y, x - 1] and not visited[y, x - 1]:
                visited[y, x - 1] = 1
                stack.append((y, x - 1))
            if x + 1 < width and binary_mask[y, x + 1] and not visited[y, x + 1]:
                visited[y, x + 1] = 1
                stack.append((y, x + 1))
        components.append((y_min, x_min, y_max, x_max, area))

    components.sort(key=lambda item: item[-1], reverse=True)
    return components


def compute_square_crop(
    bbox: tuple[int, int, int, int],
    image_hw: tuple[int, int],
    crop_min_size: int,
    crop_expand_ratio: float,
) -> tuple[int, int, int, int]:
    image_h, image_w = image_hw
    y_min, x_min, y_max, x_max = bbox
    box_h = y_max - y_min + 1
    box_w = x_max - x_min + 1
    side = max(box_h, box_w)
    side = int(math.ceil(max(float(crop_min_size), float(side) * float(crop_expand_ratio))))
    cy = 0.5 * (y_min + y_max)
    cx = 0.5 * (x_min + x_max)
    top = int(round(cy - side / 2))
    left = int(round(cx - side / 2))
    top = max(0, min(top, image_h - side))
    left = max(0, min(left, image_w - side))
    bottom = min(image_h, top + side)
    right = min(image_w, left + side)
    return top, left, bottom, right


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8), mode="RGB")


def binary_mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lesion-centric crop data for local lesion editing.")
    parser.add_argument("--metadata", type=str, default="./data/merged/diabetic/autodl/metadata_train_full_quality_maskqc_clean_colorstyle8.jsonl")
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./data/derived/lesion_crops")
    parser.add_argument("--output_size", type=int, default=512)
    parser.add_argument("--crop_min_size", type=int, default=192)
    parser.add_argument("--crop_expand_ratio", type=float, default=2.75)
    parser.add_argument("--min_pixels", type=int, default=3)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--max_components_per_class", type=int, default=6)
    parser.add_argument("--require_positive_caption", type=int, default=1)
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

    records = load_jsonl(metadata_path)
    if args.max_records > 0:
        records = records[: args.max_records]

    image_dir = out_dir / "images"
    fusion_mask_dir = out_dir / "fusion_masks"
    lesion_mask_dir = out_dir / "lesion_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    fusion_mask_dir.mkdir(parents=True, exist_ok=True)
    lesion_mask_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    stageb_fusion_rows: list[dict] = []
    stageb_lesion_rows: list[dict] = []
    class_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()

    for record_idx, record in enumerate(records):
        image_abs = Path(read_general2(record.get("image"), str(root_dir)))
        mask_abs = Path(read_general2(record.get("mask"), str(root_dir)))
        if not image_abs.exists() or not mask_abs.exists():
            continue

        image = Image.open(image_abs).convert("RGB")
        fusion_mask = Image.open(mask_abs).convert("RGB")
        mask_w, mask_h = fusion_mask.size
        image = image.resize((mask_w, mask_h), resample=Image.Resampling.BILINEAR)

        image_arr = np.asarray(image, dtype=np.uint8)
        fusion_arr = np.asarray(fusion_mask, dtype=np.uint8)

        for lesion_name in LESION_CLASS_NAMES:
            color = np.asarray(LESION_COLORS[lesion_name], dtype=np.uint8)
            binary = np.all(fusion_arr == color, axis=-1)
            components = connected_components(binary)
            if not components:
                continue

            caption_field_text = parse_caption_field(record.get("caption", ""), LESION_FIELD_MAP[lesion_name])
            if int(args.require_positive_caption) == 1 and not caption_supports_lesion(caption_field_text):
                continue
            kept = 0
            for comp_idx, (y_min, x_min, y_max, x_max, area) in enumerate(components, start=1):
                if area < args.min_pixels:
                    continue
                crop_top, crop_left, crop_bottom, crop_right = compute_square_crop(
                    bbox=(y_min, x_min, y_max, x_max),
                    image_hw=(mask_h, mask_w),
                    crop_min_size=args.crop_min_size,
                    crop_expand_ratio=args.crop_expand_ratio,
                )

                image_crop = image_arr[crop_top:crop_bottom, crop_left:crop_right]
                fusion_crop = fusion_arr[crop_top:crop_bottom, crop_left:crop_right]
                lesion_crop = binary[crop_top:crop_bottom, crop_left:crop_right]

                if image_crop.size == 0 or fusion_crop.size == 0:
                    continue

                crop_id = f"{record['id']}__{lesion_name}__c{comp_idx:02d}"
                rel_image = Path("images") / lesion_name / f"{crop_id}.png"
                rel_fusion_mask = Path("fusion_masks") / lesion_name / f"{crop_id}.png"
                rel_lesion_mask = Path("lesion_masks") / lesion_name / f"{crop_id}.png"

                abs_image_out = out_dir / rel_image
                abs_fusion_out = out_dir / rel_fusion_mask
                abs_lesion_out = out_dir / rel_lesion_mask
                abs_image_out.parent.mkdir(parents=True, exist_ok=True)
                abs_fusion_out.parent.mkdir(parents=True, exist_ok=True)
                abs_lesion_out.parent.mkdir(parents=True, exist_ok=True)

                Image.fromarray(image_crop, mode="RGB").resize(
                    (args.output_size, args.output_size),
                    resample=Image.Resampling.BILINEAR,
                ).save(abs_image_out)
                mask_to_rgb(fusion_crop).resize(
                    (args.output_size, args.output_size),
                    resample=Image.Resampling.NEAREST,
                ).save(abs_fusion_out)
                binary_mask_to_image(lesion_crop).resize(
                    (args.output_size, args.output_size),
                    resample=Image.Resampling.NEAREST,
                ).save(abs_lesion_out)

                vessel_pixels = int(np.all(fusion_crop == np.asarray((0, 255, 0), dtype=np.uint8), axis=-1).sum())
                disc_pixels = int(np.all(fusion_crop == np.asarray((255, 105, 180), dtype=np.uint8), axis=-1).sum())
                crop_pixels = int((crop_bottom - crop_top) * (crop_right - crop_left))

                vessel_ratio_in_crop = float(vessel_pixels / max(crop_pixels, 1))
                optic_disc_present_in_crop = bool(disc_pixels > 0)
                local_caption = build_local_caption(
                    lesion_name,
                    caption_field_text,
                    vessel_ratio_in_crop=vessel_ratio_in_crop,
                    optic_disc_present_in_crop=optic_disc_present_in_crop,
                )

                manifest.append(
                    {
                        "crop_id": crop_id,
                        "source_id": record["id"],
                        "lesion_class": lesion_name,
                        "component_rank": comp_idx,
                        "component_pixels": int(area),
                        "crop_box_xyxy_mask_space": [crop_left, crop_top, crop_right - 1, crop_bottom - 1],
                        "component_box_xyxy_mask_space": [x_min, y_min, x_max, y_max],
                        "caption_field_text": caption_field_text,
                        "caption": record.get("caption", ""),
                        "local_caption": local_caption,
                        "image": str(rel_image),
                        "fusion_mask": str(rel_fusion_mask),
                        "lesion_mask": str(rel_lesion_mask),
                        "source_image": record.get("image"),
                        "source_mask": record.get("mask"),
                        "output_size": args.output_size,
                        "mask_canvas_size": [mask_w, mask_h],
                        "vessel_ratio_in_crop": vessel_ratio_in_crop,
                        "optic_disc_present_in_crop": optic_disc_present_in_crop,
                    }
                )

                common_stageb_row = {
                    "id": crop_id,
                    "image": str(rel_image),
                    "caption": local_caption,
                    "caption_source": "lesion_crop_local_caption",
                    "mask_name": rel_fusion_mask.name,
                    "image_size": [args.output_size, args.output_size],
                    "mask_size": [args.output_size, args.output_size],
                    "lesion_class": lesion_name,
                    "source_id": record["id"],
                    "source_image": record.get("image"),
                    "source_mask": record.get("mask"),
                    "component_pixels": int(area),
                    "component_rank": comp_idx,
                    "caption_field_text": caption_field_text,
                    "vessel_ratio_in_crop": vessel_ratio_in_crop,
                    "optic_disc_present_in_crop": optic_disc_present_in_crop,
                }
                stageb_fusion_rows.append(
                    {
                        **common_stageb_row,
                        "mask": str(rel_fusion_mask),
                        "mask_mode": "fusion_crop",
                    }
                )
                stageb_lesion_rows.append(
                    {
                        **common_stageb_row,
                        "mask": str(rel_lesion_mask),
                        "mask_name": rel_lesion_mask.name,
                        "mask_mode": "lesion_binary",
                    }
                )
                class_counter[lesion_name] += 1
                source_counter[record["id"]] += 1
                kept += 1
                if args.max_components_per_class > 0 and kept >= args.max_components_per_class:
                    break

        if (record_idx + 1) % 500 == 0:
            print(
                json.dumps(
                    {
                        "scanned_records": record_idx + 1,
                        "crops_so_far": len(manifest),
                        "class_counter": dict(class_counter),
                    },
                    ensure_ascii=False,
                )
            )

    save_jsonl(out_dir / "manifest.jsonl", manifest)
    save_jsonl(out_dir / "metadata_stageb_fusioncrop.jsonl", stageb_fusion_rows)
    save_jsonl(out_dir / "metadata_stageb_lesionmask.jsonl", stageb_lesion_rows)
    report = {
        "metadata": str(metadata_path),
        "root_dir": str(root_dir),
        "out_dir": str(out_dir),
        "records_scanned": len(records),
        "crops_built": len(manifest),
        "stageb_fusion_rows": len(stageb_fusion_rows),
        "stageb_lesion_rows": len(stageb_lesion_rows),
        "unique_source_images": len(source_counter),
        "class_counter": dict(class_counter),
        "output_size": args.output_size,
        "crop_min_size": args.crop_min_size,
        "crop_expand_ratio": args.crop_expand_ratio,
        "min_pixels": args.min_pixels,
        "max_components_per_class": args.max_components_per_class,
        "require_positive_caption": int(args.require_positive_caption),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
