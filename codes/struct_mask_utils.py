from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image


# Full fusion mask palette observed in the diabetic dataset.
# The four lesion colors are mapped according to the user's confirmed semantics:
# crimson -> hemorrhages
# light green -> soft exudates
# cyan -> exudates
# gold -> microaneurysms
FUSION_MASK_CHANNELS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("vessels", (0, 255, 0)),
    ("optic_disc", (255, 105, 180)),
    ("hemorrhages", (220, 20, 60)),
    ("soft_exudates", (144, 238, 144)),
    ("exudates", (0, 176, 240)),
    ("microaneurysms", (255, 215, 0)),
)

DEFAULT_FUSION_STRUCT_MASK_CHANNELS = len(FUSION_MASK_CHANNELS)
FUSION_PARSER_NUM_CLASSES = DEFAULT_FUSION_STRUCT_MASK_CHANNELS + 1
_COLOR_TO_INDEX = {color: idx for idx, (_, color) in enumerate(FUSION_MASK_CHANNELS)}
_OPTIC_DISC_COLOR = np.asarray((255, 105, 180), dtype=np.uint8)


def get_struct_mask_channel_names(struct_mask_channels: int) -> list[str]:
    if struct_mask_channels == 1:
        return ["binary_mask"]
    if struct_mask_channels == DEFAULT_FUSION_STRUCT_MASK_CHANNELS:
        return [name for name, _ in FUSION_MASK_CHANNELS]
    raise ValueError(f"Unsupported struct_mask_channels={struct_mask_channels}")


def get_fusion_parser_class_names() -> list[str]:
    return ["background"] + [name for name, _ in FUSION_MASK_CHANNELS]


def make_empty_struct_mask(image_size: int, struct_mask_channels: int) -> torch.Tensor:
    return torch.zeros((struct_mask_channels, image_size, image_size), dtype=torch.float32)


def _load_grayscale_mask(mask_path: Path, image_size: int) -> torch.Tensor:
    mask_image = Image.open(mask_path).convert("L")
    mask_image = mask_image.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    arr = np.asarray(mask_image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_fusion_mask(mask_path: Path, image_size: int) -> torch.Tensor:
    mask_image = Image.open(mask_path).convert("RGB")
    mask_image = mask_image.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    arr = np.asarray(mask_image, dtype=np.uint8)

    unique_colors = {
        tuple(int(v) for v in color)
        for color in np.unique(arr.reshape(-1, 3), axis=0).tolist()
    }
    unknown_colors = sorted(color for color in unique_colors if color != (0, 0, 0) and color not in _COLOR_TO_INDEX)
    if unknown_colors:
        raise ValueError(
            "Fusion mask contains colors outside the known palette. "
            f"First unknown colors: {unknown_colors[:8]}"
        )

    mask = np.zeros((DEFAULT_FUSION_STRUCT_MASK_CHANNELS, image_size, image_size), dtype=np.float32)
    for idx, (_, color) in enumerate(FUSION_MASK_CHANNELS):
        hit = np.all(arr == np.asarray(color, dtype=np.uint8), axis=-1)
        if hit.any():
            mask[idx, hit] = 1.0
    return torch.from_numpy(mask)


def _fusion_mask_to_class_map(arr: np.ndarray) -> np.ndarray:
    class_map = np.zeros(arr.shape[:2], dtype=np.int64)
    for idx, (_, color) in enumerate(FUSION_MASK_CHANNELS, start=1):
        hit = np.all(arr == np.asarray(color, dtype=np.uint8), axis=-1)
        if hit.any():
            class_map[hit] = idx
    return class_map


def load_fusion_mask_class_map(mask_path: str | Path, image_size: int) -> torch.Tensor:
    mask_path = Path(mask_path)
    mask_image = Image.open(mask_path).convert("RGB")
    mask_image = mask_image.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    arr = np.asarray(mask_image, dtype=np.uint8)

    unique_colors = {
        tuple(int(v) for v in color)
        for color in np.unique(arr.reshape(-1, 3), axis=0).tolist()
    }
    unknown_colors = sorted(color for color in unique_colors if color != (0, 0, 0) and color not in _COLOR_TO_INDEX)
    if unknown_colors:
        raise ValueError(
            "Fusion mask contains colors outside the known palette. "
            f"First unknown colors: {unknown_colors[:8]}"
        )

    return torch.from_numpy(_fusion_mask_to_class_map(arr))


def load_struct_mask_tensor(
    mask_path: str | Path,
    image_size: int,
    struct_mask_channels: int = 1,
) -> torch.Tensor:
    mask_path = Path(mask_path)
    if struct_mask_channels == 1:
        return _load_grayscale_mask(mask_path, image_size)
    if struct_mask_channels == DEFAULT_FUSION_STRUCT_MASK_CHANNELS:
        return _load_fusion_mask(mask_path, image_size)
    raise ValueError(
        "Unsupported struct_mask_channels. "
        f"Expected 1 or {DEFAULT_FUSION_STRUCT_MASK_CHANNELS}, got {struct_mask_channels}"
    )


def infer_laterality_from_name(path_or_name: str | Path | None) -> str | None:
    if path_or_name is None:
        return None
    stem = Path(path_or_name).stem.lower()
    if stem.endswith("_left"):
        return "left eye"
    if stem.endswith("_right"):
        return "right eye"
    return None


def infer_optic_disc_side(mask_path: str | Path | None) -> str | None:
    geometry = infer_optic_disc_geometry(mask_path)
    if geometry is None:
        return None
    return geometry["side"]


def infer_optic_disc_geometry(mask_path: str | Path | None) -> dict[str, float | str] | None:
    if mask_path is None:
        return None

    mask_path = Path(mask_path)
    if not mask_path.exists():
        return None

    arr = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    disc = np.all(arr == _OPTIC_DISC_COLOR, axis=-1)
    if not disc.any():
        return None

    ys, xs = np.where(disc)
    x_norm = float(xs.mean()) / max(float(arr.shape[1] - 1), 1.0)
    y_norm = float(ys.mean()) / max(float(arr.shape[0] - 1), 1.0)
    if x_norm < 0.4:
        side = "left side"
    elif x_norm > 0.6:
        side = "right side"
    else:
        side = "center"

    edge_margin = float(
        min(
            x_norm,
            1.0 - x_norm,
            y_norm,
            1.0 - y_norm,
        )
    )
    return {
        "side": side,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "edge_margin": edge_margin,
    }


def describe_horizontal_position(x_norm: float) -> str:
    if x_norm < 0.12:
        return "very close to the left edge"
    if x_norm < 0.28:
        return "on the left side"
    if x_norm < 0.42:
        return "slightly left of center"
    if x_norm <= 0.58:
        return "near the horizontal center"
    if x_norm <= 0.72:
        return "slightly right of center"
    if x_norm <= 0.88:
        return "on the right side"
    return "very close to the right edge"


def describe_vertical_position(y_norm: float) -> str:
    if y_norm < 0.24:
        return "high in the frame"
    if y_norm < 0.42:
        return "slightly above the horizontal midline"
    if y_norm <= 0.58:
        return "near the vertical center"
    if y_norm <= 0.76:
        return "slightly below the horizontal midline"
    return "low in the frame"


def describe_edge_margin(edge_margin: float) -> str:
    if edge_margin < 0.08:
        return "with a very tight margin from the image boundary"
    if edge_margin < 0.14:
        return "with a narrow but visible margin from the image boundary"
    if edge_margin < 0.22:
        return "with a moderate margin from the image boundary"
    return "with a comfortable margin from the image boundary"


def _extract_prompt_section(text: str, section_name: str) -> tuple[str | None, str]:
    pattern = re.compile(
        rf"(?:(?<=^)|(?<=\s))({re.escape(section_name)}:\s.*?)(?=\s+[A-Z_]+:|$)",
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        return None, " ".join(text.split())
    section = " ".join(match.group(1).split())
    remaining = " ".join((text[: match.start(1)] + " " + text[match.end(1) :]).split())
    return section, remaining


def augment_caption_with_struct_hints(
    caption: str,
    image_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    struct_mask_channels: int = 1,
) -> str:
    base_caption = " ".join((caption or "").split())
    color_style, caption_without_color_style = _extract_prompt_section(base_caption, "COLOR_STYLE")

    if "LATERALITY:" in base_caption or "ANATOMY_HINT:" in base_caption:
        parts = []
        if color_style is not None:
            parts.append(color_style)
        if caption_without_color_style:
            parts.append(caption_without_color_style)
        return " ".join(parts)

    prefix: list[str] = []

    laterality = infer_laterality_from_name(image_path or mask_path)
    if laterality is not None:
        prefix.append(f"LATERALITY: {laterality}.")

    if int(struct_mask_channels) == DEFAULT_FUSION_STRUCT_MASK_CHANNELS:
        disc_geometry = infer_optic_disc_geometry(mask_path)
        if disc_geometry is not None:
            disc_side = str(disc_geometry["side"])
            horizontal_desc = describe_horizontal_position(float(disc_geometry["x_norm"]))
            vertical_desc = describe_vertical_position(float(disc_geometry["y_norm"]))
            margin_desc = describe_edge_margin(float(disc_geometry["edge_margin"]))
            prefix.append(f"ANATOMY_HINT: optic disc is on the {disc_side} of the image.")
            prefix.append(f"ANATOMY_HINT: optic disc center is {horizontal_desc} and {vertical_desc}.")
            prefix.append(f"ANATOMY_HINT: optic disc remains fully inside the fundus boundary {margin_desc}.")
            if disc_side != "center":
                prefix.append(
                    f"ANATOMY_HINT: major retinal vessels radiate from the optic disc on the {disc_side} of the image."
                )

    body_caption = caption_without_color_style if color_style is not None else base_caption

    if not prefix:
        parts = []
        if color_style is not None:
            parts.append(color_style)
        if body_caption:
            parts.append(body_caption)
        return " ".join(parts)
    if not body_caption and color_style is None:
        return " ".join(prefix)

    parts = []
    if color_style is not None:
        parts.append(color_style)
    parts.extend(prefix)
    if body_caption:
        parts.append(body_caption)
    return " ".join(parts)
