import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from PIL import Image
from torchvision import models, transforms


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(path: Path) -> list[Path]:
    images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found under {path}")
    return images


def inception_feature_model(device: torch.device) -> torch.nn.Module:
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def load_batch(paths: list[Path], transform: transforms.Compose) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path).convert("RGB") as image:
            images.append(transform(image))
    return torch.stack(images, dim=0)


@torch.inference_mode()
def extract_features(image_dir: Path, batch_size: int, device: torch.device) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = inception_feature_model(device)
    paths = list_images(image_dir)
    feats = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = load_batch(batch_paths, transform).to(device)
        output = model(batch)
        if hasattr(output, "logits"):
            output = output.logits
        feats.append(output.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(feats, axis=0)


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    diff = mu_real - mu_fake
    covmean = scipy.linalg.sqrtm(sigma_real @ sigma_fake)
    if not np.isfinite(covmean).all():
        eps = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma_real + eps) @ (sigma_fake + eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(fid)


def polynomial_mmd2_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    dim = x.shape[1]
    k_xx = ((x @ x.T) / dim + 1.0) ** 3
    k_yy = ((y @ y.T) / dim + 1.0) ** 3
    k_xy = ((x @ y.T) / dim + 1.0) ** 3
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    m = x.shape[0]
    n = y.shape[0]
    if m < 2 or n < 2:
        raise ValueError("KID requires at least two images in each directory.")
    return float(k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (n * (n - 1)) - 2.0 * k_xy.mean())


def calculate_kid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    num_subsets: int,
    max_subset_size: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    subset_size = min(max_subset_size, real_features.shape[0], fake_features.shape[0])
    values = []
    for _ in range(num_subsets):
        real_idx = rng.choice(real_features.shape[0], subset_size, replace=False)
        fake_idx = rng.choice(fake_features.shape[0], subset_size, replace=False)
        values.append(polynomial_mmd2_unbiased(real_features[real_idx], fake_features[fake_idx]))
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID and KID for two image folders.")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_subsets", type=int, default=100)
    parser.add_argument("--max_subset_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    real_dir = Path(args.real_dir).expanduser().resolve()
    fake_dir = Path(args.fake_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_features = extract_features(real_dir, args.batch_size, device)
    fake_features = extract_features(fake_dir, args.batch_size, device)
    fid = calculate_fid(real_features, fake_features)
    kid_mean, kid_std = calculate_kid(
        real_features,
        fake_features,
        num_subsets=args.num_subsets,
        max_subset_size=args.max_subset_size,
        seed=args.seed,
    )
    result = {
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "real_count": int(real_features.shape[0]),
        "fake_count": int(fake_features.shape[0]),
        "feature_dim": int(real_features.shape[1]),
        "fid": fid,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "kid_num_subsets": args.num_subsets,
        "kid_subset_size": min(args.max_subset_size, real_features.shape[0], fake_features.shape[0]),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
