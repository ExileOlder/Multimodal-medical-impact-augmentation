from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FundusGeometry:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    radius: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.center_x, self.center_y)


def _foreground_mask(image: Image.Image, threshold: int = 10) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return arr.mean(axis=2) > int(threshold)


def detect_fundus_geometry(
    image: Image.Image,
    *,
    threshold: int = 10,
    percentile: float = 99.5,
) -> FundusGeometry:
    mask = _foreground_mask(image, threshold=threshold)
    ys, xs = np.nonzero(mask)
    width, height = image.size
    if len(xs) == 0:
        radius = min(width, height) / 2.0
        return FundusGeometry(
            center_x=(width - 1) / 2.0,
            center_y=(height - 1) / 2.0,
            radius_x=radius,
            radius_y=radius,
            radius=radius,
        )

    center_x = float(xs.mean())
    center_y = float(ys.mean())
    radius_x = max(float(np.percentile(np.abs(xs - center_x), percentile)), 1.0)
    radius_y = max(float(np.percentile(np.abs(ys - center_y), percentile)), 1.0)
    radius = max(radius_x, radius_y)
    return FundusGeometry(
        center_x=center_x,
        center_y=center_y,
        radius_x=radius_x,
        radius_y=radius_y,
        radius=radius,
    )


def extract_square_crop(
    image: Image.Image,
    *,
    center: tuple[float, float],
    side: int,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    image = image.convert("RGB")
    side = max(1, int(side))
    cx, cy = center

    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side
    bottom = top + side

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(image.size[0], right)
    src_bottom = min(image.size[1], bottom)

    canvas = Image.new("RGB", (side, side), color=background)
    if src_right <= src_left or src_bottom <= src_top:
        return canvas

    region = image.crop((src_left, src_top, src_right, src_bottom))
    paste_x = src_left - left
    paste_y = src_top - top
    canvas.paste(region, (paste_x, paste_y))
    return canvas


def apply_circular_aperture(
    image: Image.Image,
    *,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    cx = (width - 1) / 2.0 if center is None else float(center[0])
    cy = (height - 1) / 2.0 if center is None else float(center[1])
    if radius is None:
        radius = min(width, height) / 2.0
    radius = max(float(radius), 1.0)

    yy, xx = np.ogrid[:height, :width]
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2

    arr = np.asarray(image, dtype=np.uint8).copy()
    arr[~circle] = np.asarray(background, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def canonicalize_fundus_image(
    image: Image.Image,
    *,
    output_size: int,
    threshold: int = 10,
    percentile: float = 99.5,
    padding_ratio: float = 0.01,
    background: tuple[int, int, int] = (0, 0, 0),
    mask_outside_circle: bool = True,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    image = image.convert("RGB")
    geometry = detect_fundus_geometry(
        image,
        threshold=threshold,
        percentile=percentile,
    )
    crop_side = max(1, int(round(2.0 * geometry.radius * (1.0 + 2.0 * float(padding_ratio)))))
    square = extract_square_crop(
        image,
        center=geometry.center,
        side=crop_side,
        background=background,
    )
    square = square.resize((int(output_size), int(output_size)), resample=resample)

    if mask_outside_circle:
        circle_radius = (float(output_size) * 0.5) / max(1.0 + 2.0 * float(padding_ratio), 1e-6)
        square = apply_circular_aperture(square, radius=circle_radius, background=background)
    return square


def place_square_into_fundus_canvas(
    image: Image.Image,
    reference_size: tuple[int, int],
    *,
    resample: Image.Resampling,
    fundus_center: tuple[float, float],
    fundus_radius: float,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    image = image.convert("RGB")
    ref_w, ref_h = reference_size
    if ref_w <= 0 or ref_h <= 0:
        raise ValueError(f"Invalid reference size: {reference_size}")

    diameter = max(1, int(round(float(fundus_radius) * 2.0)))
    diameter = min(diameter, max(ref_w, ref_h))
    resized = image.resize((diameter, diameter), resample=resample)

    canvas = Image.new("RGB", (ref_w, ref_h), color=background)
    cx, cy = fundus_center
    offset_x = int(round(cx - diameter / 2.0))
    offset_y = int(round(cy - diameter / 2.0))
    canvas.paste(resized, (offset_x, offset_y))
    return canvas
