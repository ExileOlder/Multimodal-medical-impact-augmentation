import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights
from transformers import CLIPModel, CLIPProcessor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.image_paths[index]).convert("RGB") as image:
            return self.transform(image)


def collect_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    paths = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No images were found under: {image_dir}")
    return paths


def load_inception(device: torch.device, inception_weights_path: str) -> nn.Module:
    if inception_weights_path:
        model = models.inception_v3(weights=None, transform_input=False, aux_logits=False)
        state_dict = torch.load(inception_weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        try:
            model = models.inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
                aux_logits=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Inception weights. "
                "If the server is offline, provide --inception_weights_path with a local .pth file."
            ) from exc
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


def get_inception_features(
    image_paths: list[Path],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        ImagePathDataset(image_paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    features: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = model(batch)
            features.append(feats.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    mu_r = np.mean(real_feats, axis=0)
    mu_f = np.mean(fake_feats, axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_f = np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean)
    return float(fid)


def polynomial_mmd(x: np.ndarray, y: np.ndarray) -> float:
    dim = x.shape[1]
    gamma = 1.0 / dim
    k_xx = (gamma * x @ x.T + 1.0) ** 3
    k_yy = (gamma * y @ y.T + 1.0) ** 3
    k_xy = (gamma * x @ y.T + 1.0) ** 3
    n = x.shape[0]
    m = y.shape[0]
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    term_xx = k_xx.sum() / (n * (n - 1))
    term_yy = k_yy.sum() / (m * (m - 1))
    term_xy = 2.0 * k_xy.mean()
    return float(term_xx + term_yy - term_xy)


def compute_kid(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    subset_size: int,
    num_subsets: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    subset_size = min(subset_size, len(real_feats), len(fake_feats))
    if subset_size < 2:
        raise ValueError("At least two images are required to compute KID.")

    values: list[float] = []
    for _ in range(num_subsets):
        real_idx = rng.choice(len(real_feats), size=subset_size, replace=False)
        fake_idx = rng.choice(len(fake_feats), size=subset_size, replace=False)
        values.append(polynomial_mmd(real_feats[real_idx], fake_feats[fake_idx]))
    return float(np.mean(values)), float(np.std(values))


def load_prompts(prompt_file: Path, num_images: int) -> list[str]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompts = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in: {prompt_file}")
    if len(prompts) == 1:
        return prompts * num_images
    if len(prompts) != num_images:
        raise ValueError(
            f"Prompt count ({len(prompts)}) does not match generated image count ({num_images}). "
            "Use one prompt for all images or one prompt per image."
        )
    return prompts


def compute_clip_score(
    image_paths: list[Path],
    prompts: list[str],
    clip_model_path: str,
    device: torch.device,
    batch_size: int,
) -> dict:
    model_source = clip_model_path or "openai/clip-vit-base-patch32"
    try:
        processor = CLIPProcessor.from_pretrained(model_source)
        model = CLIPModel.from_pretrained(model_source).eval().to(device)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load CLIP model. If the server is offline, provide --clip_model_path with a local model directory."
        ) from exc

    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(text=batch_prompts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            batch_scores = torch.sum(image_embeds * text_embeds, dim=-1)
            scores.extend(batch_scores.detach().cpu().tolist())
            for image in images:
                image.close()

    return {
        "clip_score_mean": float(np.mean(scores)),
        "clip_score_std": float(np.std(scores)),
        "clip_score_min": float(np.min(scores)),
        "clip_score_max": float(np.max(scores)),
    }


def parse_runtime(runtime_file: Path, num_images: int) -> dict:
    if not runtime_file.exists():
        raise FileNotFoundError(f"Runtime file not found: {runtime_file}")
    content = runtime_file.read_text(encoding="utf-8").strip()
    total_seconds = float(content)
    avg_seconds = total_seconds / num_images
    return {
        "total_runtime_seconds": total_seconds,
        "avg_runtime_seconds_per_image": avg_seconds,
        "images_per_second": num_images / total_seconds if total_seconds > 0 else math.inf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID, KID, CLIP Score, and inference-time metrics.")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory of generated images")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory of real reference images")
    parser.add_argument("--prompt_file", type=str, required=True, help="Prompt file used for generation")
    parser.add_argument("--runtime_file", type=str, required=True, help="File containing total runtime seconds")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save metric results")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KID subsets")
    parser.add_argument("--kid_subset_size", type=int, default=50, help="Subset size for KID")
    parser.add_argument("--kid_subsets", type=int, default=20, help="Number of subsets for KID")
    parser.add_argument("--clip_model_path", type=str, default="", help="Local CLIP model directory")
    parser.add_argument("--inception_weights_path", type=str, default="", help="Local Inception v3 .pth path")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    generated_dir = Path(args.generated_dir).expanduser().resolve()
    real_dir = Path(args.real_dir).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    runtime_file = Path(args.runtime_file).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    fake_images = collect_images(generated_dir)
    real_images = collect_images(real_dir)

    if len(real_images) < 2 or len(fake_images) < 2:
        raise ValueError("FID/KID require image sets, not a single real image or a single generated image.")

    inception = load_inception(device, args.inception_weights_path)
    real_feats = get_inception_features(real_images, inception, device, args.batch_size)
    fake_feats = get_inception_features(fake_images, inception, device, args.batch_size)
    fid = compute_fid(real_feats, fake_feats)
    kid_mean, kid_std = compute_kid(
        real_feats,
        fake_feats,
        subset_size=args.kid_subset_size,
        num_subsets=args.kid_subsets,
        seed=args.seed,
    )

    prompts = load_prompts(prompt_file, len(fake_images))
    clip_metrics = compute_clip_score(fake_images, prompts, args.clip_model_path, device, args.batch_size)
    runtime_metrics = parse_runtime(runtime_file, len(fake_images))

    results = {
        "generated_dir": str(generated_dir),
        "real_dir": str(real_dir),
        "num_generated_images": len(fake_images),
        "num_real_images": len(real_images),
        "fid": fid,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
    }
    results.update(clip_metrics)
    results.update(runtime_metrics)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
