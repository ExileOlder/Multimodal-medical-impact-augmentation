from __future__ import annotations

import argparse
import json
import logging
import math
import random
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from data.data_reader import read_general2
from fundus_geometry import canonicalize_fundus_image
from models.fusion_parser import FusionMaskParser
from struct_mask_utils import (
    FUSION_MASK_CHANNELS,
    FUSION_PARSER_NUM_CLASSES,
    get_fusion_parser_class_names,
    load_fusion_mask_class_map,
)


LESION_CLASS_NAMES = {"hemorrhages", "soft_exudates", "exudates", "microaneurysms"}
# Keep lesion indices in the same order as class names returned by the parser.
LESION_CLASS_INDICES = [
    idx for idx, class_name in enumerate(get_fusion_parser_class_names()) if class_name in LESION_CLASS_NAMES
]
LESION_FOCUS_CROP_FRACTIONS = {
    "hemorrhages": (0.22, 0.40),
    "soft_exudates": (0.24, 0.44),
    "exudates": (0.20, 0.38),
    "microaneurysms": (0.16, 0.30),
}
LESION_FOCUS_MIN_CROP_PX = {
    "hemorrhages": 112,
    "soft_exudates": 128,
    "exudates": 112,
    "microaneurysms": 96,
}


def create_logger(log_path: Path) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    return logging.getLogger("fusion_parser")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


def analyze_class_distribution(
    records: list[dict],
    root_dir: Path,
    image_size: int,
    logger: logging.Logger,
    stats_image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    pixel_counts = torch.zeros(FUSION_PARSER_NUM_CLASSES, dtype=torch.float64)
    record_presence = torch.zeros(FUSION_PARSER_NUM_CLASSES, dtype=torch.float64)
    sample_class_ids: list[list[int]] = []

    for idx, record in enumerate(records, start=1):
        mask_path = read_general2(record["mask"], str(root_dir))
        class_map = load_fusion_mask_class_map(mask_path, stats_image_size)
        bincount = torch.bincount(class_map.view(-1), minlength=FUSION_PARSER_NUM_CLASSES).to(torch.float64)
        pixel_counts += bincount
        present_ids = torch.nonzero(bincount[1:] > 0, as_tuple=False).flatten() + 1
        if present_ids.numel() > 0:
            record_presence[present_ids] += 1.0
            sample_class_ids.append([int(v) for v in present_ids.tolist()])
        else:
            sample_class_ids.append([])
        if idx % 2000 == 0:
            logger.info(f"class-stats scan: {idx}/{len(records)}")

    return pixel_counts, record_presence, sample_class_ids


def build_class_weights(
    pixel_counts: torch.Tensor,
    *,
    background_ce_weight: float,
    max_fg_ce_weight: float,
    lesion_ce_boost: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_pixels = pixel_counts.sum().clamp_min(1.0)
    pixel_freq = pixel_counts / total_pixels
    fg_freq = pixel_freq[1:].clamp_min(1e-8)
    fg_weight = torch.rsqrt(fg_freq)
    fg_weight = fg_weight / fg_weight.mean().clamp_min(1e-8)
    fg_weight = fg_weight.clamp(min=1.0, max=max_fg_ce_weight)
    fg_weight[2:] = (fg_weight[2:] * lesion_ce_boost).clamp(max=max_fg_ce_weight)

    ce_weight = torch.ones(FUSION_PARSER_NUM_CLASSES, dtype=torch.float32)
    ce_weight[0] = background_ce_weight
    ce_weight[1:] = fg_weight.float()
    dice_weight = ce_weight[1:] / ce_weight[1:].sum().clamp_min(1e-8)
    return ce_weight, dice_weight


def build_sample_weights(
    sample_class_ids: list[list[int]],
    ce_weight: torch.Tensor,
    *,
    lesion_sample_boost: float,
    lesion_focus_class_weight: dict[int, float] | None = None,
) -> torch.Tensor:
    weights = []
    ce_weight = ce_weight.cpu()
    for class_ids in sample_class_ids:
        if not class_ids:
            weights.append(1.0)
            continue
        weight = max(float(ce_weight[class_id].item()) for class_id in class_ids)
        lesion_class_ids = [class_id for class_id in class_ids if class_id in LESION_CLASS_INDICES]
        if lesion_class_ids:
            focus_boost = 1.0
            if lesion_focus_class_weight is not None:
                focus_boost = max(lesion_focus_class_weight.get(class_id, 1.0) for class_id in lesion_class_ids)
            weight *= lesion_sample_boost * focus_boost
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.double)


def build_lesion_focus_class_weights(
    pixel_counts: torch.Tensor,
    record_presence: torch.Tensor,
    total_records: int,
) -> dict[int, float]:
    total_pixels = pixel_counts.sum().clamp_min(1.0)
    pixel_freq = pixel_counts / total_pixels
    record_freq = record_presence / max(total_records, 1)
    lesion_focus_weight: dict[int, float] = {}
    raw_weights = []
    for class_idx in LESION_CLASS_INDICES:
        pix_term = float(pixel_freq[class_idx].clamp_min(1e-8).pow(-0.35).item())
        rec_term = float(record_freq[class_idx].clamp_min(1e-8).pow(-0.50).item())
        raw_weights.append((class_idx, pix_term * rec_term))
    mean_raw = sum(weight for _, weight in raw_weights) / max(len(raw_weights), 1)
    for class_idx, raw_weight in raw_weights:
        lesion_focus_weight[class_idx] = float(max(1.0, min(raw_weight / max(mean_raw, 1e-8), 4.0)))
    return lesion_focus_weight


class FusionParserDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        root_dir: Path,
        image_size: int,
        *,
        training: bool = False,
        lesion_focus_prob: float = 0.0,
        focus_crop_min_frac: float = 0.35,
        focus_crop_max_frac: float = 0.7,
        lesion_focus_class_weight: dict[int, float] | None = None,
        fundus_geometry_align: bool = False,
        fundus_padding_ratio: float = 0.01,
    ) -> None:
        self.records = records
        self.root_dir = root_dir
        self.image_size = image_size
        self.training = training
        self.lesion_focus_prob = lesion_focus_prob
        self.focus_crop_min_frac = focus_crop_min_frac
        self.focus_crop_max_frac = focus_crop_max_frac
        self.lesion_focus_class_weight = lesion_focus_class_weight or {}
        self.fundus_geometry_align = bool(fundus_geometry_align)
        self.fundus_padding_ratio = float(fundus_padding_ratio)
        transform_ops = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
        self.image_transform = transforms.Compose(transform_ops)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _rgb_mask_to_class_map(mask_rgb: np.ndarray) -> np.ndarray:
        class_map = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        for idx, (name, color) in enumerate(FUSION_MASK_CHANNELS, start=1):
            if name not in LESION_CLASS_NAMES and name not in {"vessels", "optic_disc"}:
                continue
            hit = np.all(mask_rgb == np.asarray(color, dtype=np.uint8), axis=-1)
            if hit.any():
                class_map[hit] = idx
        return class_map

    def _maybe_focus_crop(
        self,
        image: Image.Image,
        mask_rgb: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        if not self.training or self.lesion_focus_prob <= 0.0 or random.random() >= self.lesion_focus_prob:
            return image, mask_rgb

        lesion_masks: list[tuple[int, str, np.ndarray, int]] = []
        for idx, (name, color) in enumerate(FUSION_MASK_CHANNELS, start=1):
            if name not in LESION_CLASS_NAMES:
                continue
            hit = np.all(mask_rgb == np.asarray(color, dtype=np.uint8), axis=-1)
            if hit.any():
                lesion_masks.append((idx, name, hit, int(hit.sum())))
        if not lesion_masks:
            return image, mask_rgb

        lesion_weights = []
        for class_idx, class_name, lesion_mask, lesion_area in lesion_masks:
            focus_weight = self.lesion_focus_class_weight.get(class_idx, 1.0)
            within_image_small_target_bias = 1.0 / max(float(lesion_area) ** 0.25, 1.0)
            lesion_weights.append(focus_weight * within_image_small_target_bias)
        chosen_idx, chosen_name, chosen_mask, _ = random.choices(
            lesion_masks,
            weights=lesion_weights,
            k=1,
        )[0]
        ys, xs = np.nonzero(chosen_mask)
        if len(xs) == 0:
            return image, mask_rgb

        point_idx = random.randrange(len(xs))
        cx = int(xs[point_idx])
        cy = int(ys[point_idx])
        width, height = image.size
        side = min(width, height)
        crop_min_frac, crop_max_frac = LESION_FOCUS_CROP_FRACTIONS.get(
            chosen_name,
            (self.focus_crop_min_frac, self.focus_crop_max_frac),
        )
        crop_frac = random.uniform(crop_min_frac, crop_max_frac)
        min_crop_size = LESION_FOCUS_MIN_CROP_PX.get(chosen_name, 128)
        crop_size = max(int(side * crop_frac), min_crop_size)
        crop_size = min(crop_size, width, height)
        half = crop_size // 2
        left = max(0, min(cx - half, width - crop_size))
        top = max(0, min(cy - half, height - crop_size))
        right = left + crop_size
        bottom = top + crop_size
        return image.crop((left, top, right, bottom)), mask_rgb[top:bottom, left:right]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_path = read_general2(record["image"], str(self.root_dir))
        mask_path = read_general2(record["mask"], str(self.root_dir))
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")
        # Some source images and fusion masks are not pixel-aligned on disk.
        # Canonicalize the real image into the square fundus field-of-view
        # before any focused crop so image and mask share one geometry space.
        if self.fundus_geometry_align:
            image = canonicalize_fundus_image(
                image,
                output_size=mask_image.size[0],
                padding_ratio=self.fundus_padding_ratio,
            )
        elif image.size != mask_image.size:
            image = image.resize(mask_image.size, resample=Image.Resampling.BILINEAR)
        mask_rgb = np.asarray(mask_image, dtype=np.uint8)
        image, mask_rgb = self._maybe_focus_crop(image, mask_rgb)
        image_tensor = self.image_transform(image)
        if image.size == (self.image_size, self.image_size):
            class_map = torch.from_numpy(self._rgb_mask_to_class_map(mask_rgb)).clone()
        else:
            mask_image = Image.fromarray(mask_rgb, mode="RGB").resize(
                (self.image_size, self.image_size), resample=Image.Resampling.NEAREST
            )
            class_map = torch.from_numpy(self._rgb_mask_to_class_map(np.asarray(mask_image, dtype=np.uint8))).clone()
        return image_tensor.contiguous(), class_map.long().contiguous()


def fusion_parser_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([image.contiguous() for image, _ in batch], dim=0)
    targets = torch.stack([target.contiguous() for _, target in batch], dim=0)
    return images, targets


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        collate_fn=fusion_parser_collate_fn,
    )


def dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    fg_dice = dice[1:]
    if class_weight is None:
        return 1.0 - fg_dice.mean()
    fg_weight = class_weight.to(logits.device)
    fg_weight = fg_weight / fg_weight.sum().clamp_min(1e-8)
    return 1.0 - (fg_dice * fg_weight).sum()


def compute_class_dice(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    metrics = {}
    class_names = get_fusion_parser_class_names()
    for cls_idx, cls_name in enumerate(class_names):
        pred_mask = pred == cls_idx
        tgt_mask = target == cls_idx
        intersection = (pred_mask & tgt_mask).sum().item()
        denom = pred_mask.sum().item() + tgt_mask.sum().item()
        if denom == 0:
            metrics[cls_name] = float("nan")
        else:
            metrics[cls_name] = float((2.0 * intersection + 1e-6) / (denom + 1e-6))
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_weight: torch.Tensor,
    dice_weight: torch.Tensor,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_count = 0
    dice_buckets = {name: [] for name in get_fusion_parser_class_names()}
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        ce = F.cross_entropy(logits, target, weight=ce_weight)
        dice = dice_loss_from_logits(logits, target, FUSION_PARSER_NUM_CLASSES, class_weight=dice_weight)
        loss = ce + dice
        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)
        batch_dice = compute_class_dice(logits, target, FUSION_PARSER_NUM_CLASSES)
        for key, val in batch_dice.items():
            if not math.isnan(val):
                dice_buckets[key].append(val)

    summary = {
        "loss": total_loss / max(total_count, 1),
        "dice": {key: (sum(vals) / len(vals) if vals else float("nan")) for key, vals in dice_buckets.items()},
    }
    fg_vals = [v for k, v in summary["dice"].items() if k != "background" and not math.isnan(v)]
    summary["mean_fg_dice"] = sum(fg_vals) / max(len(fg_vals), 1)
    fg_names = get_fusion_parser_class_names()[1:]
    lesion_names = [name for name in fg_names if name in LESION_CLASS_NAMES]
    weighted_score = 0.0
    total_weight = 0.0
    for name, weight in zip(fg_names, dice_weight.tolist()):
        val = summary["dice"].get(name, float("nan"))
        if not math.isnan(val):
            weighted_score += float(weight) * float(val)
            total_weight += float(weight)
    summary["weighted_fg_dice"] = weighted_score / max(total_weight, 1e-8)

    lesion_vals = [summary["dice"].get(name, float("nan")) for name in lesion_names]
    lesion_vals = [val for val in lesion_vals if not math.isnan(val)]
    summary["mean_lesion_dice"] = sum(lesion_vals) / max(len(lesion_vals), 1)

    lesion_weighted_score = 0.0
    lesion_total_weight = 0.0
    for name, weight in zip(fg_names, dice_weight.tolist()):
        if name not in LESION_CLASS_NAMES:
            continue
        val = summary["dice"].get(name, float("nan"))
        if not math.isnan(val):
            lesion_weighted_score += float(weight) * float(val)
            lesion_total_weight += float(weight)
    summary["weighted_lesion_dice"] = lesion_weighted_score / max(lesion_total_weight, 1e-8)
    summary["selection_score"] = 0.35 * summary["weighted_fg_dice"] + 0.65 * summary["weighted_lesion_dice"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 6-class fusion-mask parser.")
    parser.add_argument("--metadata", type=str, default="./data/merged/diabetic/autodl/metadata_train_full_quality_clean.jsonl")
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results/train/fusion_mask_parser")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_records", type=int, default=0)
    parser.add_argument("--max_val_records", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--background_ce_weight", type=float, default=0.02)
    parser.add_argument("--max_fg_ce_weight", type=float, default=16.0)
    parser.add_argument("--lesion_ce_boost", type=float, default=2.5)
    parser.add_argument("--lesion_sample_boost", type=float, default=2.0)
    parser.add_argument("--use_weighted_sampler", type=int, default=1)
    parser.add_argument("--lesion_focus_prob", type=float, default=0.6)
    parser.add_argument("--focus_crop_min_frac", type=float, default=0.35)
    parser.add_argument("--focus_crop_max_frac", type=float, default=0.7)
    parser.add_argument("--stats_image_size", type=int, default=256)
    parser.add_argument("--fundus_geometry_align", action="store_true", default=False)
    parser.add_argument("--fundus_padding_ratio", type=float, default=0.01)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    code_root = Path(__file__).resolve().parent
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = (code_root / args.metadata).resolve()
    root_dir = Path(args.root_dir)
    if not root_dir.is_absolute():
        root_dir = (code_root / args.root_dir).resolve()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (code_root / args.results_dir).resolve()
    run_dir = results_dir / datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_fusion_parser")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(run_dir / "train.log")

    records = load_jsonl(metadata_path)
    train_records, val_records = split_records(records, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_train_records > 0:
        train_records = train_records[: args.max_train_records]
    if args.max_val_records > 0:
        val_records = val_records[: args.max_val_records]

    logger.info(f"train_records={len(train_records)} val_records={len(val_records)} device={device}")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    pixel_counts, record_presence, sample_class_ids = analyze_class_distribution(
        train_records,
        root_dir=root_dir,
        image_size=args.image_size,
        logger=logger,
        stats_image_size=min(args.stats_image_size, args.image_size),
    )
    ce_weight_cpu, dice_weight_cpu = build_class_weights(
        pixel_counts,
        background_ce_weight=args.background_ce_weight,
        max_fg_ce_weight=args.max_fg_ce_weight,
        lesion_ce_boost=args.lesion_ce_boost,
    )
    lesion_focus_class_weight = build_lesion_focus_class_weights(
        pixel_counts,
        record_presence,
        total_records=len(train_records),
    )
    class_names = get_fusion_parser_class_names()
    logger.info(
        "train class pixel ratio="
        + json.dumps(
            {
                class_names[idx]: float(pixel_counts[idx].item() / pixel_counts.sum().clamp_min(1.0).item())
                for idx in range(FUSION_PARSER_NUM_CLASSES)
            },
            ensure_ascii=False,
        )
    )
    logger.info(
        "train class record presence="
        + json.dumps(
            {
                class_names[idx]: int(record_presence[idx].item())
                for idx in range(1, FUSION_PARSER_NUM_CLASSES)
            },
            ensure_ascii=False,
        )
    )
    logger.info(
        "ce_weight="
        + json.dumps({class_names[idx]: float(ce_weight_cpu[idx].item()) for idx in range(FUSION_PARSER_NUM_CLASSES)}, ensure_ascii=False)
    )
    logger.info(
        "lesion_focus_class_weight="
        + json.dumps(
            {
                class_names[class_idx]: float(lesion_focus_class_weight[class_idx])
                for class_idx in LESION_CLASS_INDICES
            },
            ensure_ascii=False,
        )
    )

    train_dataset = FusionParserDataset(
        train_records,
        root_dir=root_dir,
        image_size=args.image_size,
        training=True,
        lesion_focus_prob=args.lesion_focus_prob,
        focus_crop_min_frac=args.focus_crop_min_frac,
        focus_crop_max_frac=args.focus_crop_max_frac,
        lesion_focus_class_weight=lesion_focus_class_weight,
        fundus_geometry_align=args.fundus_geometry_align,
        fundus_padding_ratio=args.fundus_padding_ratio,
    )
    val_dataset = FusionParserDataset(
        val_records,
        root_dir=root_dir,
        image_size=args.image_size,
        training=False,
        fundus_geometry_align=args.fundus_geometry_align,
        fundus_padding_ratio=args.fundus_padding_ratio,
    )
    sampler = None
    if int(args.use_weighted_sampler) == 1:
        sample_weights = build_sample_weights(
            sample_class_ids,
            ce_weight_cpu,
            lesion_sample_boost=args.lesion_sample_boost,
            lesion_focus_class_weight=lesion_focus_class_weight,
        )
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info(
            f"weighted sampler enabled: min={sample_weights.min().item():.4f} "
            f"mean={sample_weights.mean().item():.4f} max={sample_weights.max().item():.4f}"
        )
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True, sampler=sampler)
    val_loader = build_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    model = FusionMaskParser(num_classes=FUSION_PARSER_NUM_CLASSES, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    ce_weight = ce_weight_cpu.to(device)
    dice_weight = dice_weight_cpu.to(device)

    best_selection_score = -1.0
    global_step = 0
    train_iter = iter(train_loader)
    running_loss = 0.0

    while global_step < args.max_steps:
        try:
            images, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, target = next(train_iter)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            logits = model(images)
            ce = F.cross_entropy(logits, target, weight=ce_weight)
            dice = dice_loss_from_logits(logits, target, FUSION_PARSER_NUM_CLASSES, class_weight=dice_weight)
            loss = ce + dice
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        running_loss += loss.item()

        if global_step % args.log_every == 0:
            logger.info(
                f"step={global_step:05d}/{args.max_steps} "
                f"loss={running_loss / args.log_every:.4f} ce={ce.item():.4f} dice={dice.item():.4f}"
            )
            running_loss = 0.0

        if global_step % args.val_every == 0 or global_step == args.max_steps:
            summary = evaluate(model, val_loader, device, ce_weight, dice_weight)
            logger.info(
                f"[val] step={global_step:05d} loss={summary['loss']:.4f} "
                f"mean_fg_dice={summary['mean_fg_dice']:.4f} "
                f"weighted_fg_dice={summary['weighted_fg_dice']:.4f} "
                f"mean_lesion_dice={summary['mean_lesion_dice']:.4f} "
                f"weighted_lesion_dice={summary['weighted_lesion_dice']:.4f} "
                f"selection_score={summary['selection_score']:.4f} "
                f"dice={summary['dice']}"
            )
            ckpt = {
                "model": model.state_dict(),
                "step": global_step,
                "args": vars(args),
                "val": summary,
            }
            torch.save(ckpt, run_dir / "parser_last.pt")
            if summary["selection_score"] > best_selection_score:
                best_selection_score = summary["selection_score"]
                torch.save(ckpt, run_dir / "parser_best.pt")
                logger.info(
                    f"[best] step={global_step:05d} "
                    f"selection_score={best_selection_score:.4f} "
                    f"weighted_lesion_dice={summary['weighted_lesion_dice']:.4f}"
                )

    logger.info("done")


if __name__ == "__main__":
    main()
