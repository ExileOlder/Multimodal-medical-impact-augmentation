from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data.data_reader import read_general2
from models.fusion_parser import FusionMaskParser
from struct_mask_utils import FUSION_MASK_CHANNELS, get_fusion_parser_class_names
from train_fusion_parser import FusionParserDataset, load_jsonl, split_records


BACKGROUND_COLOR = (0, 0, 0)
CLASS_NAMES = get_fusion_parser_class_names()
CLASS_COLORS = [BACKGROUND_COLOR] + [color for _, color in FUSION_MASK_CHANNELS]
LESION_NAMES = {"hemorrhages", "soft_exudates", "exudates", "microaneurysms"}


def class_map_to_rgb(class_map: np.ndarray) -> Image.Image:
    rgb = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(CLASS_COLORS):
        rgb[class_map == class_idx] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def load_resized_source_image(record: dict, root_dir: Path, image_size: int) -> Image.Image:
    image_path = Path(read_general2(record["image"], str(root_dir)))
    image = Image.open(image_path).convert("RGB")
    return image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)


def blend_overlay(image: Image.Image, mask_rgb: Image.Image, alpha: float = 0.42) -> Image.Image:
    return Image.blend(image.convert("RGB"), mask_rgb.convert("RGB"), alpha=alpha)


def record_has_any_lesion(record: dict, root_dir: Path) -> bool:
    mask_path = Path(read_general2(record["mask"], str(root_dir)))
    arr = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    for lesion_name, color in FUSION_MASK_CHANNELS:
        if lesion_name not in LESION_NAMES:
            continue
        if np.all(arr == np.asarray(color, dtype=np.uint8), axis=-1).any():
            return True
    return False


def choose_indices(records: list[dict], root_dir: Path, limit: int, prefer_lesion: bool) -> list[int]:
    indices = list(range(len(records)))
    if not prefer_lesion:
        return indices[:limit]
    lesion_indices = [idx for idx, record in enumerate(records) if record_has_any_lesion(record, root_dir)]
    if len(lesion_indices) >= limit:
        return lesion_indices[:limit]
    merged = lesion_indices + [idx for idx in indices if idx not in set(lesion_indices)]
    return merged[:limit]


def get_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        return ImageFont.load_default()


def render_sheet(
    dataset: FusionParserDataset,
    records: list[dict],
    root_dir: Path,
    model: FusionMaskParser,
    device: torch.device,
    output_path: Path,
    *,
    indices: list[int],
    panel_size: int = 256,
) -> None:
    columns = ["Original", "GT Mask", "Pred Mask", "Pred Overlay"]
    font = get_font()
    row_label_w = 180
    header_h = 34
    cell_pad = 8
    panel_w = panel_size + cell_pad * 2
    canvas_w = row_label_w + panel_w * len(columns)
    canvas_h = header_h + len(indices) * (panel_size + cell_pad * 2)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col_idx, col_name in enumerate(columns):
        x = row_label_w + col_idx * panel_w + cell_pad
        draw.text((x, 8), col_name, fill=(0, 0, 0), font=font)

    for row_idx, dataset_idx in enumerate(indices):
        record = records[dataset_idx]
        image_tensor, target_tensor = dataset[dataset_idx]
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            pred_tensor = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        gt_tensor = target_tensor.cpu().numpy().astype(np.uint8)
        source_image = load_resized_source_image(record, root_dir, dataset.image_size)
        gt_rgb = class_map_to_rgb(gt_tensor)
        pred_rgb = class_map_to_rgb(pred_tensor)
        overlay = blend_overlay(source_image, pred_rgb)

        panels = [
            source_image.resize((panel_size, panel_size), resample=Image.Resampling.BILINEAR),
            gt_rgb.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST),
            pred_rgb.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST),
            overlay.resize((panel_size, panel_size), resample=Image.Resampling.BILINEAR),
        ]

        row_y = header_h + row_idx * (panel_size + cell_pad * 2)
        caption = f"{record['id']}"
        draw.text((12, row_y + 10), caption, fill=(0, 0, 0), font=font)

        for col_idx, panel in enumerate(panels):
            x = row_label_w + col_idx * panel_w + cell_pad
            y = row_y + cell_pad
            canvas.paste(panel, (x, y))
            draw.rectangle(
                (x - 1, y - 1, x + panel_size, y + panel_size),
                outline=(180, 180, 180),
                width=1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render comparison sheets for a trained fusion parser.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="parser_best.pt")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--panel_size", type=int, default=256)
    parser.add_argument("--prefer_lesion", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_name", type=str, default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())

    metadata_path = Path(config["metadata"])
    root_dir = Path(config["root_dir"])
    if not metadata_path.is_absolute():
        metadata_path = (run_dir.parents[3] / metadata_path).resolve()
    if not root_dir.is_absolute():
        root_dir = (run_dir.parents[3] / root_dir).resolve()

    records = load_jsonl(metadata_path)
    _, val_records = split_records(records, val_ratio=config["val_ratio"], seed=config["seed"])
    if config.get("max_val_records", 0) > 0:
        val_records = val_records[: config["max_val_records"]]

    dataset = FusionParserDataset(
        val_records,
        root_dir=root_dir,
        image_size=config["image_size"],
        training=False,
    )
    ckpt = torch.load(run_dir / args.ckpt_name, map_location=args.device)
    model = FusionMaskParser(num_classes=len(CLASS_NAMES), base_channels=config["base_channels"])
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    indices = choose_indices(
        val_records,
        root_dir=root_dir,
        limit=args.limit,
        prefer_lesion=bool(int(args.prefer_lesion)),
    )
    output_name = args.output_name or f"{Path(args.ckpt_name).stem}_comparison_sheet.png"
    render_sheet(
        dataset,
        val_records,
        root_dir,
        model,
        torch.device(args.device),
        run_dir / output_name,
        indices=indices,
        panel_size=args.panel_size,
    )
    print(str(run_dir / output_name))


if __name__ == "__main__":
    main()
