from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fundus_geometry as fg


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(root_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (root_dir / path).resolve()


def fit_panel(
    image: Image.Image,
    panel_size: int,
    *,
    resample: Image.Resampling,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    scale = min(panel_size / src_w, panel_size / src_h)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = image.resize((dst_w, dst_h), resample=resample)

    canvas = Image.new("RGB", (panel_size, panel_size), color=background)
    offset_x = (panel_size - dst_w) // 2
    offset_y = (panel_size - dst_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_sheet(
    records: list[dict],
    root_dir: Path,
    generated_dir: Path,
    output_path: Path,
    panel_size: int,
    max_items: int,
    generated_label: str,
    align_mode: str,
    fundus_padding_ratio: float,
) -> int:
    label_font = load_font(max(18, panel_size // 18))
    id_font = load_font(max(16, panel_size // 22))
    rows: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []

    for record in records:
        record_id = str(record["id"])
        image_path = resolve_path(root_dir, record.get("image") or record.get("image_path"))
        mask_path = resolve_path(root_dir, record.get("mask") or record.get("mask_path"))
        generated_path = generated_dir / f"{record_id}.png"
        if not image_path or not mask_path:
            continue
        if not image_path.exists() or not mask_path.exists() or not generated_path.exists():
            continue
        source_image = Image.open(image_path).convert("RGB")
        if align_mode == "canonical_square":
            real_image = fit_panel(
                fg.canonicalize_fundus_image(
                    source_image,
                    output_size=panel_size,
                    padding_ratio=fundus_padding_ratio,
                ),
                panel_size,
                resample=Image.Resampling.BILINEAR,
            )
            mask_image = fit_panel(
                Image.open(mask_path).convert("RGB"),
                panel_size,
                resample=Image.Resampling.NEAREST,
            )
            generated_image = fit_panel(
                Image.open(generated_path).convert("RGB"),
                panel_size,
                resample=Image.Resampling.BILINEAR,
            )
        else:
            reference_size = source_image.size
            fundus_geometry = fg.detect_fundus_geometry(source_image)
            fundus_center = fundus_geometry.center
            fundus_radius = float(fundus_geometry.radius)
            real_image = fit_panel(
                source_image,
                panel_size,
                resample=Image.Resampling.BILINEAR,
            )
            mask_image = fit_panel(
                fg.place_square_into_fundus_canvas(
                    Image.open(mask_path).convert("RGB"),
                    reference_size,
                    resample=Image.Resampling.NEAREST,
                    fundus_center=fundus_center,
                    fundus_radius=fundus_radius,
                ),
                panel_size,
                resample=Image.Resampling.NEAREST,
            )
            generated_image = fit_panel(
                fg.place_square_into_fundus_canvas(
                    Image.open(generated_path).convert("RGB"),
                    reference_size,
                    resample=Image.Resampling.BILINEAR,
                    fundus_center=fundus_center,
                    fundus_radius=fundus_radius,
                ),
                panel_size,
                resample=Image.Resampling.BILINEAR,
            )
        rows.append((record_id, real_image, mask_image, generated_image))
        if len(rows) >= max_items:
            break

    if not rows:
        raise RuntimeError("No paired real/mask/generated samples found.")

    cols = 3
    gutter = 18
    left_pad = 24
    top_pad = 54
    row_gap = 48
    width = left_pad * 2 + cols * panel_size + (cols - 1) * gutter
    height = top_pad + len(rows) * (panel_size + row_gap)
    canvas = Image.new("RGB", (width, height), color=(10, 10, 10))
    draw = ImageDraw.Draw(canvas)

    labels = ("Real", "Mask", generated_label)
    for col_idx, label in enumerate(labels):
        x = left_pad + col_idx * (panel_size + gutter)
        draw.text((x, 16), label, fill=(240, 240, 240), font=label_font)

    for row_idx, (record_id, real_image, mask_image, generated_image) in enumerate(rows):
        y = top_pad + row_idx * (panel_size + row_gap)
        draw.text((left_pad, y - 28), record_id, fill=(220, 220, 220), font=id_font)
        for col_idx, image in enumerate((real_image, mask_image, generated_image)):
            x = left_pad + col_idx * (panel_size + gutter)
            canvas.paste(image, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Real/Mask/Generated comparison sheets.")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--panel_size", type=int, default=256)
    parser.add_argument("--max_items", type=int, default=8)
    parser.add_argument("--generated_label", type=str, default="Generated")
    parser.add_argument(
        "--align_mode",
        type=str,
        choices=["canonical_square", "fundus_canvas"],
        default="canonical_square",
    )
    parser.add_argument("--fundus_padding_ratio", type=float, default=0.01)
    args = parser.parse_args()

    metadata = Path(args.metadata).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()
    generated_dir = Path(args.generated_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    records = load_jsonl(metadata)
    count = make_sheet(
        records=records,
        root_dir=root_dir,
        generated_dir=generated_dir,
        output_path=output_path,
        panel_size=int(args.panel_size),
        max_items=int(args.max_items),
        generated_label=str(args.generated_label),
        align_mode=str(args.align_mode),
        fundus_padding_ratio=float(args.fundus_padding_ratio),
    )
    print(f"[ok] saved {output_path} with {count} rows")


if __name__ == "__main__":
    main()
