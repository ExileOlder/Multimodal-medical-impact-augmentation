import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from fundus_geometry import canonicalize_fundus_image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_metadata(path: Path) -> list[dict]:
    records = []
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
    return path if path.is_absolute() else (root_dir / path).resolve()


def find_generated_path(generated_dir: Path, record_id: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = generated_dir / f"{record_id}{ext}"
        if candidate.exists():
            return candidate
    matches = [p for p in generated_dir.rglob("*") if p.is_file() and p.stem == record_id and p.suffix.lower() in IMAGE_EXTS]
    return sorted(matches)[0] if matches else None


def load_rgb(
    path: Path,
    size: tuple[int, int] | None = None,
    *,
    canonicalize: bool = False,
    fundus_padding_ratio: float = 0.01,
) -> np.ndarray:
    with Image.open(path).convert("RGB") as image:
        if canonicalize:
            if size is None:
                raise ValueError("size is required when canonicalize=True")
            image = canonicalize_fundus_image(
                image,
                output_size=size[1],
                padding_ratio=fundus_padding_ratio,
            )
        elif size is not None:
            image = image.resize(size, Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path).convert("L") as mask:
        if mask.size != size:
            mask = mask.resize(size, Image.NEAREST)
        arr = np.asarray(mask, dtype=np.float32) / 255.0
    return arr


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def psnr_from_mse(value: float) -> float:
    if value <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / value))


def to_gray(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    denom = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    if denom == 0:
        return 1.0
    return float(((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / denom)


def masked_metric_arrays(real: np.ndarray, fake: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_bool = mask > 0.5
    if mask_bool.sum() == 0:
        return np.empty((0, real.shape[-1]), dtype=np.float32), np.empty((0, fake.shape[-1]), dtype=np.float32)
    real_pixels = real[mask_bool]
    fake_pixels = fake[mask_bool]
    return real_pixels, fake_pixels


def build_fundus_mask(arr: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got {arr.shape}")
    return arr.mean(axis=2) > threshold


def masked_rgb_stats(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = arr[mask]
    if pixels.size == 0:
        zeros = np.zeros((3,), dtype=np.float32)
        return zeros, zeros
    return pixels.mean(axis=0), pixels.std(axis=0)


def chroma_from_rgb_mean(rgb_mean: np.ndarray) -> np.ndarray:
    denom = float(np.clip(rgb_mean.sum(), 1e-6, None))
    return rgb_mean / denom


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired structural consistency for generated fundus images.")
    parser.add_argument("--metadata", type=str, required=True, help="metadata_val.jsonl or metadata_train.jsonl")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory containing generated images named by record id")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory used by metadata relative paths")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save structural metrics")
    parser.add_argument("--topk", type=int, default=20, help="Number of worst-case examples to record")
    parser.add_argument(
        "--alignment_mode",
        type=str,
        choices=["canonical_square", "raw_resize"],
        default="canonical_square",
        help="Compare in canonical square fundus space or by resizing everything to the raw image size.",
    )
    parser.add_argument(
        "--fundus_padding_ratio",
        type=float,
        default=0.01,
        help="Extra margin around the detected fundus circle when alignment_mode=canonical_square.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    generated_dir = Path(args.generated_dir).expanduser().resolve()
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    records = load_metadata(metadata_path)
    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    global_l1_values = []
    global_mse_values = []
    global_psnr_values = []
    global_ssim_values = []
    mask_l1_values = []
    mask_mse_values = []
    mask_psnr_values = []
    mask_ssim_values = []
    mask_coverage_values = []
    fundus_mean_abs_error_values = []
    fundus_std_abs_error_values = []
    fundus_chroma_l1_values = []
    fundus_rb_gap_abs_error_values = []
    worst_cases = []
    missing_generated = []

    for record in records:
        record_id = str(record["id"])
        generated_path = find_generated_path(generated_dir, record_id)
        if generated_path is None:
            missing_generated.append(record_id)
            continue

        real_path = resolve_path(root_dir, record["image"])
        mask_path = resolve_path(root_dir, record["mask"])

        if args.alignment_mode == "canonical_square":
            fake = load_rgb(generated_path)
            target_size = (fake.shape[1], fake.shape[0])
            real = load_rgb(
                real_path,
                size=target_size,
                canonicalize=True,
                fundus_padding_ratio=args.fundus_padding_ratio,
            )
            mask = load_mask(mask_path, size=target_size)
        else:
            real = load_rgb(real_path)
            fake = load_rgb(generated_path, size=(real.shape[1], real.shape[0]))
            mask = load_mask(mask_path, size=(real.shape[1], real.shape[0]))

        g_l1 = l1(real, fake)
        g_mse = mse(real, fake)
        g_psnr = psnr_from_mse(g_mse)
        g_ssim = ssim_gray(to_gray(real), to_gray(fake))

        real_masked, fake_masked = masked_metric_arrays(real, fake, mask)
        coverage = float((mask > 0.5).mean())

        if real_masked.size == 0:
            m_l1 = 0.0
            m_mse = 0.0
            m_psnr = float("inf")
            m_ssim = 1.0
        else:
            m_l1 = l1(real_masked, fake_masked)
            m_mse = mse(real_masked, fake_masked)
            m_psnr = psnr_from_mse(m_mse)
            m_ssim = ssim_gray(real_masked.mean(axis=-1), fake_masked.mean(axis=-1))

        global_l1_values.append(g_l1)
        global_mse_values.append(g_mse)
        global_psnr_values.append(g_psnr)
        global_ssim_values.append(g_ssim)
        mask_l1_values.append(m_l1)
        mask_mse_values.append(m_mse)
        mask_psnr_values.append(m_psnr)
        mask_ssim_values.append(m_ssim)
        mask_coverage_values.append(coverage)

        fundus_mask = build_fundus_mask(real)
        real_fundus_mean, real_fundus_std = masked_rgb_stats(real, fundus_mask)
        fake_fundus_mean, fake_fundus_std = masked_rgb_stats(fake, fundus_mask)
        real_fundus_chroma = chroma_from_rgb_mean(real_fundus_mean)
        fake_fundus_chroma = chroma_from_rgb_mean(fake_fundus_mean)
        real_rb_gap = float(real_fundus_mean[0] - real_fundus_mean[2])
        fake_rb_gap = float(fake_fundus_mean[0] - fake_fundus_mean[2])

        fundus_mean_abs_error = float(np.abs(real_fundus_mean - fake_fundus_mean).mean())
        fundus_std_abs_error = float(np.abs(real_fundus_std - fake_fundus_std).mean())
        fundus_chroma_l1 = float(np.abs(real_fundus_chroma - fake_fundus_chroma).mean())
        fundus_rb_gap_abs_error = float(abs(real_rb_gap - fake_rb_gap))

        fundus_mean_abs_error_values.append(fundus_mean_abs_error)
        fundus_std_abs_error_values.append(fundus_std_abs_error)
        fundus_chroma_l1_values.append(fundus_chroma_l1)
        fundus_rb_gap_abs_error_values.append(fundus_rb_gap_abs_error)

        worst_cases.append(
            {
                "id": record_id,
                "generated_path": str(generated_path),
                "global_l1": g_l1,
                "global_ssim": g_ssim,
                "mask_l1": m_l1,
                "mask_ssim": m_ssim,
                "mask_coverage": coverage,
                "fundus_mean_abs_error": fundus_mean_abs_error,
                "fundus_std_abs_error": fundus_std_abs_error,
                "fundus_chroma_l1": fundus_chroma_l1,
                "fundus_rb_gap_abs_error": fundus_rb_gap_abs_error,
                "real_fundus_rgb_mean": [float(x) for x in real_fundus_mean.tolist()],
                "fake_fundus_rgb_mean": [float(x) for x in fake_fundus_mean.tolist()],
            }
        )

    worst_cases_by_mask_ssim = sorted(worst_cases, key=lambda item: (item["mask_ssim"], -item["mask_l1"]))[: args.topk]
    worst_cases_by_color_drift = sorted(
        worst_cases,
        key=lambda item: (item["fundus_chroma_l1"], item["fundus_rb_gap_abs_error"]),
        reverse=True,
    )[: args.topk]

    result = {
        "metadata": str(metadata_path),
        "generated_dir": str(generated_dir),
        "root_dir": str(root_dir),
        "alignment_mode": args.alignment_mode,
        "paired_records": len(global_l1_values),
        "missing_generated_count": len(missing_generated),
        "missing_generated_sample": missing_generated[:50],
        "global_l1": summarize(global_l1_values),
        "global_mse": summarize(global_mse_values),
        "global_psnr": summarize(global_psnr_values),
        "global_ssim": summarize(global_ssim_values),
        "mask_l1": summarize(mask_l1_values),
        "mask_mse": summarize(mask_mse_values),
        "mask_psnr": summarize(mask_psnr_values),
        "mask_ssim": summarize(mask_ssim_values),
        "mask_coverage_ratio": summarize(mask_coverage_values),
        "fundus_mean_abs_error": summarize(fundus_mean_abs_error_values),
        "fundus_std_abs_error": summarize(fundus_std_abs_error_values),
        "fundus_chroma_l1": summarize(fundus_chroma_l1_values),
        "fundus_rb_gap_abs_error": summarize(fundus_rb_gap_abs_error_values),
        "worst_cases_by_mask_ssim": worst_cases_by_mask_ssim,
        "worst_cases_by_color_drift": worst_cases_by_color_drift,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved structural metrics to {output_json}")


if __name__ == "__main__":
    main()
