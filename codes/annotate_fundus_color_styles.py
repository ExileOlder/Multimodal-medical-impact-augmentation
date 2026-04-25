from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fundus_geometry import canonicalize_fundus_image
from struct_mask_utils import FUSION_MASK_CHANNELS


@dataclass(frozen=True)
class RecordFeature:
    record_id: str
    image_path: Path
    features: np.ndarray
    stats: dict[str, float]


@dataclass(frozen=True)
class FeatureTask:
    image_path: Path
    mask_path: Path | None
    record_id: str


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_path(root_dir: Path, value: str | None) -> Path:
    if not value:
        raise ValueError("Missing image path in metadata record.")
    path = Path(value)
    return path if path.is_absolute() else (root_dir / value).resolve()


def build_radius_map(size: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, num=size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.sqrt(xx ** 2 + yy ** 2)


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros((arr.shape[-1],), dtype=np.float32)
    return arr[mask].mean(axis=0).astype(np.float32)


def masked_scalar_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(arr[mask].mean())


def masked_scalar_std(arr: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(arr[mask].std())


def masked_quantile(arr: np.ndarray, mask: np.ndarray, q: float) -> float:
    if not mask.any():
        return 0.0
    return float(np.quantile(arr[mask], q))


def select_region_mask(base_mask: np.ndarray, region_mask: np.ndarray, min_pixels: int = 64) -> np.ndarray:
    selected = base_mask & region_mask
    if selected.sum() >= min_pixels:
        return selected
    if base_mask.sum() >= min_pixels:
        return base_mask
    return region_mask


def mean_rgb_stats(arr: np.ndarray, mask: np.ndarray, prefix: str, stats: dict[str, float]) -> None:
    mean_rgb = masked_mean(arr, mask)
    stats[f"{prefix}_r"] = float(mean_rgb[0])
    stats[f"{prefix}_g"] = float(mean_rgb[1])
    stats[f"{prefix}_b"] = float(mean_rgb[2])


def load_fusion_regions(mask_path: Path | None, image_size: int) -> dict[str, np.ndarray]:
    if mask_path is None or not mask_path.exists():
        return {}
    arr = np.asarray(
        Image.open(mask_path).convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    regions: dict[str, np.ndarray] = {}
    for name, color in FUSION_MASK_CHANNELS:
        regions[name] = np.all(arr == np.asarray(color, dtype=np.uint8), axis=-1)
    return regions


def compute_record_feature(
    record: dict,
    *,
    root_dir: Path,
    image_size: int,
    padding_ratio: float,
    radius_map: np.ndarray,
) -> RecordFeature:
    image_path = resolve_path(root_dir, record.get("image") or record.get("image_path"))
    mask_value = record.get("mask") or record.get("mask_path")
    mask_path = resolve_path(root_dir, mask_value) if mask_value else None
    image = Image.open(image_path).convert("RGB")
    image = canonicalize_fundus_image(
        image,
        output_size=image_size,
        padding_ratio=padding_ratio,
    )
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    fundus_mask = image_np.mean(axis=2) > 0.04
    if fundus_mask.sum() < 128:
        raise ValueError(f"Fundus mask too small for record {record.get('id')}")

    luma = 0.299 * image_np[..., 0] + 0.587 * image_np[..., 1] + 0.114 * image_np[..., 2]
    sat_map = image_np.max(axis=2) - image_np.min(axis=2)
    fusion_regions = load_fusion_regions(mask_path, image_size=image_size)
    lesion_mask = np.zeros_like(fundus_mask, dtype=bool)
    for name in ("hemorrhages", "soft_exudates", "exudates", "microaneurysms"):
        lesion_mask |= fusion_regions.get(name, np.zeros_like(fundus_mask, dtype=bool))
    optic_disc_mask = fusion_regions.get("optic_disc", np.zeros_like(fundus_mask, dtype=bool))
    vessel_mask = fusion_regions.get("vessels", np.zeros_like(fundus_mask, dtype=bool))

    clean_retina_mask = fundus_mask & ~(lesion_mask | optic_disc_mask)
    if clean_retina_mask.sum() < 128:
        clean_retina_mask = fundus_mask
    background_retina_mask = clean_retina_mask & ~vessel_mask
    if background_retina_mask.sum() < 128:
        background_retina_mask = clean_retina_mask

    core_mask = select_region_mask(background_retina_mask, radius_map <= 0.38)
    mid_mask = select_region_mask(background_retina_mask, (radius_map > 0.38) & (radius_map <= 0.70))
    edge_mask = select_region_mask(background_retina_mask, (radius_map > 0.70) & (radius_map <= 0.98))

    mean_rgb = masked_mean(image_np, background_retina_mask)
    std_rgb = image_np[background_retina_mask].std(axis=0).astype(np.float32)
    chroma = mean_rgb / max(float(mean_rgb.sum()), 1e-6)
    saturation = masked_scalar_mean(sat_map, background_retina_mask)
    luma_mean = masked_scalar_mean(luma, background_retina_mask)
    luma_std = masked_scalar_std(luma, background_retina_mask)
    core_luma = masked_scalar_mean(luma, core_mask)
    mid_luma = masked_scalar_mean(luma, mid_mask)
    edge_luma = masked_scalar_mean(luma, edge_mask)
    vignette_drop = max(core_luma - edge_luma, 0.0)
    core_rgb = masked_mean(image_np, core_mask)
    mid_rgb = masked_mean(image_np, mid_mask)
    edge_rgb = masked_mean(image_np, edge_mask)
    haze_score = masked_scalar_mean((1.0 - sat_map) * np.clip(luma - 0.18, 0.0, 1.0), background_retina_mask)
    warmth_score = float((mean_rgb[0] - mean_rgb[1]) + 0.5 * (mean_rgb[1] - mean_rgb[2]))
    pinkness_score = float(mean_rgb[0] + mean_rgb[2] - 2.0 * mean_rgb[1])
    green_bias = float(mean_rgb[1] - 0.5 * (mean_rgb[0] + mean_rgb[2]))
    core_green_bias = float(core_rgb[1] - 0.5 * (core_rgb[0] + core_rgb[2]))
    mid_green_bias = float(mid_rgb[1] - 0.5 * (mid_rgb[0] + mid_rgb[2]))
    edge_green_bias = float(edge_rgb[1] - 0.5 * (edge_rgb[0] + edge_rgb[2]))
    red_dominance_penalty = max(float(mean_rgb[0] - mean_rgb[1]) - 0.08, 0.0)
    green_cast_score = float(green_bias - pinkness_score + 0.6 * haze_score - red_dominance_penalty)
    luma_q15 = masked_quantile(luma, background_retina_mask, 0.15)
    luma_q50 = masked_quantile(luma, background_retina_mask, 0.50)
    luma_q85 = masked_quantile(luma, background_retina_mask, 0.85)
    sat_q50 = masked_quantile(sat_map, background_retina_mask, 0.50)
    sat_q85 = masked_quantile(sat_map, background_retina_mask, 0.85)

    features = np.asarray(
        [
            luma_mean,
            luma_std,
            saturation,
            float(chroma[0] - chroma[1]),
            float(chroma[1] - chroma[2]),
            float(chroma[0] - chroma[2]),
            float(core_rgb[0] - core_rgb[1]),
            float(core_rgb[1] - core_rgb[2]),
            float(mid_rgb[0] - mid_rgb[1]),
            float(mid_rgb[1] - mid_rgb[2]),
            float(edge_rgb[0] - edge_rgb[1]),
            float(edge_rgb[1] - edge_rgb[2]),
            vignette_drop,
            core_luma - mid_luma,
            luma_q15,
            luma_q50,
            luma_q85,
            sat_q50,
            sat_q85,
            haze_score,
            warmth_score,
            pinkness_score,
            green_cast_score,
            green_bias,
            core_green_bias,
            mid_green_bias,
            edge_green_bias,
        ],
        dtype=np.float32,
    )
    stats = {
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "std_r": float(std_rgb[0]),
        "std_g": float(std_rgb[1]),
        "std_b": float(std_rgb[2]),
        "luma_mean": luma_mean,
        "luma_std": luma_std,
        "core_luma": core_luma,
        "mid_luma": mid_luma,
        "edge_luma": edge_luma,
        "vignette_drop": vignette_drop,
        "saturation_mean": saturation,
        "haze_score": haze_score,
        "warmth_score": warmth_score,
        "pinkness_score": pinkness_score,
        "green_cast_score": green_cast_score,
        "green_bias": green_bias,
        "core_green_bias": core_green_bias,
        "mid_green_bias": mid_green_bias,
        "edge_green_bias": edge_green_bias,
        "luma_q15": luma_q15,
        "luma_q50": luma_q50,
        "luma_q85": luma_q85,
        "sat_q50": sat_q50,
        "sat_q85": sat_q85,
        "lesion_coverage": float(lesion_mask.mean()),
        "optic_disc_coverage": float(optic_disc_mask.mean()),
        "vessel_coverage": float(vessel_mask.mean()),
        "chroma_r": float(chroma[0]),
        "chroma_g": float(chroma[1]),
        "chroma_b": float(chroma[2]),
    }
    mean_rgb_stats(image_np, core_mask, "core", stats)
    mean_rgb_stats(image_np, mid_mask, "mid", stats)
    mean_rgb_stats(image_np, edge_mask, "edge", stats)
    return RecordFeature(
        record_id=str(record["id"]),
        image_path=image_path,
        features=features,
        stats=stats,
    )


def compute_record_feature_from_task(
    task: FeatureTask,
    *,
    image_size: int,
    padding_ratio: float,
) -> RecordFeature:
    radius_map = build_radius_map(int(image_size))
    record = {
        "id": task.record_id,
        "image": str(task.image_path),
        "mask": str(task.mask_path) if task.mask_path is not None else None,
    }
    return compute_record_feature(
        record,
        root_dir=Path("/"),
        image_size=image_size,
        padding_ratio=padding_ratio,
        radius_map=radius_map,
    )


def stack_features(record_features: list[RecordFeature]) -> np.ndarray:
    return np.stack([item.features for item in record_features], axis=0)


def standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (features - mean) / std, mean, std


def build_feature_weights(num_dims: int) -> np.ndarray:
    weights = np.ones((num_dims,), dtype=np.float32)
    if num_dims >= 22:
        weights[19] = 1.25
        weights[20] = 1.35
        weights[21] = 1.75
    if num_dims >= 27:
        weights[-5:] = 2.75
    return weights


def kmeans_plus_plus_init(features: np.ndarray, num_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n = features.shape[0]
    if n < num_clusters:
        raise ValueError(f"num_clusters={num_clusters} exceeds sample count {n}")
    centroids = [features[rng.integers(0, n)]]
    while len(centroids) < num_clusters:
        dist_sq = np.min(
            np.stack([np.sum((features - c) ** 2, axis=1) for c in centroids], axis=1),
            axis=1,
        )
        probs = dist_sq / np.clip(dist_sq.sum(), 1e-12, None)
        centroids.append(features[rng.choice(n, p=probs)])
    return np.stack(centroids, axis=0)


def fit_kmeans(
    features: np.ndarray,
    *,
    num_clusters: int,
    seed: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus_init(features, num_clusters, rng)
    assignments = np.zeros((features.shape[0],), dtype=np.int64)

    for _ in range(max_iter):
        dist_sq = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(dist_sq, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        new_centroids = centroids.copy()
        for idx in range(num_clusters):
            members = features[assignments == idx]
            if len(members) == 0:
                farthest_idx = int(np.argmax(np.min(dist_sq, axis=1)))
                new_centroids[idx] = features[farthest_idx]
            else:
                new_centroids[idx] = members.mean(axis=0)
        centroids = new_centroids
    return centroids, assignments


def assign_clusters(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dist_sq = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(dist_sq, axis=1)


def brightness_word(luma_mean: float) -> str:
    if luma_mean >= 0.62:
        return "bright"
    if luma_mean >= 0.49:
        return "medium-bright"
    if luma_mean >= 0.38:
        return "medium"
    return "dim"


def hue_word(stats: dict[str, float]) -> str:
    mean_r = float(stats["mean_r"])
    mean_g = float(stats["mean_g"])
    mean_b = float(stats["mean_b"])
    rg = mean_r - mean_g
    gb = mean_g - mean_b
    rb = mean_r - mean_b
    green_bias = float(stats.get("green_bias", mean_g - 0.5 * (mean_r + mean_b)))
    red_dominance_penalty = max(rg - 0.08, 0.0)
    green_cast_score = float(
        stats.get(
            "green_cast_score",
            green_bias - float(stats.get("pinkness_score", 0.0)) + 0.6 * float(stats.get("haze_score", 0.0)) - red_dominance_penalty,
        )
    )
    if green_cast_score >= 0.08:
        return "green-tinged beige"
    if green_cast_score >= 0.03 and rg < 0.10:
        return "green-tinged yellow-beige"
    if rg >= 0.16 and gb >= 0.11:
        return "orange-red"
    if rg >= 0.10 and gb >= 0.07:
        return "warm orange"
    if rg >= 0.05 and gb >= 0.03:
        return "yellow-beige"
    if rb >= 0.08 and gb < 0.03:
        return "rosy beige"
    if abs(rg) < 0.04 and abs(gb) < 0.04:
        return "neutral beige"
    return "reddish brown"


def clarity_word(stats: dict[str, float]) -> str:
    haze_score = float(stats.get("haze_score", 0.0))
    saturation_mean = float(stats.get("saturation_mean", 0.0))
    if haze_score >= 0.22 and saturation_mean <= 0.14:
        return "hazy"
    if haze_score >= 0.16:
        return "slightly hazy"
    if saturation_mean >= 0.24:
        return "clear"
    return "balanced"


def contrast_word(luma_std: float) -> str:
    if luma_std >= 0.16:
        return "strong contrast"
    if luma_std >= 0.10:
        return "moderate contrast"
    return "soft contrast"


def vignette_word(vignette_drop: float) -> str:
    if vignette_drop >= 0.16:
        return "pronounced peripheral darkening"
    if vignette_drop >= 0.08:
        return "moderate peripheral darkening"
    return "mild peripheral darkening"


def make_style_prompt(stats: dict[str, float]) -> str:
    return (
        f"overall fundus tone is {brightness_word(stats['luma_mean'])} {hue_word(stats)} "
        f"with a {clarity_word(stats)} background, {contrast_word(stats['luma_std'])} and {vignette_word(stats['vignette_drop'])}"
    )


def make_style_slug(prompt: str) -> str:
    clean = prompt.lower()
    for src, dst in [
        ("overall fundus tone is ", ""),
        (" with ", "_"),
        (" and ", "_"),
        ("-", "_"),
        (" ", "_"),
    ]:
        clean = clean.replace(src, dst)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip(" _.,")


def cluster_mean_stats(record_features: list[RecordFeature], member_indices: np.ndarray) -> dict[str, float]:
    if len(member_indices) == 0:
        raise ValueError("Cannot summarize an empty cluster.")
    keys = tuple(record_features[int(member_indices[0])].stats.keys())
    summary: dict[str, float] = {}
    for key in keys:
        values = [record_features[int(idx)].stats[key] for idx in member_indices]
        summary[key] = float(np.mean(values))
    return summary


def enrich_record(record: dict, cluster_id: int, style_code: str, style_prompt: str, style_slug: str) -> dict:
    enriched = dict(record)
    base_caption = str(record.get("caption_base") or record.get("caption") or "").strip()
    style_sentence = f"COLOR_STYLE: {style_code} {style_prompt}."
    if base_caption:
        enriched_caption = f"{style_sentence} {base_caption}"
    else:
        enriched_caption = style_sentence
    enriched["caption_base"] = base_caption
    enriched["caption"] = enriched_caption
    enriched["color_style_cluster_id"] = int(cluster_id)
    enriched["color_style_code"] = style_code
    enriched["color_style_label"] = style_slug
    enriched["color_style_prompt"] = style_prompt
    return enriched


def build_summary(
    *,
    centroids_raw: np.ndarray,
    centroids_norm: np.ndarray,
    assignments: np.ndarray,
    features_norm: np.ndarray,
    record_features: list[RecordFeature],
    style_codes: list[str],
    style_prompts: list[str],
    style_slugs: list[str],
) -> dict:
    summary_clusters = []
    for cluster_id, centroid in enumerate(centroids_raw):
        member_indices = np.where(assignments == cluster_id)[0]
        if len(member_indices) == 0:
            example_ids = []
        else:
            member_features = features_norm[member_indices]
            centroid_norm = centroids_norm[cluster_id]
            member_distances = np.sum((member_features - centroid_norm[None, :]) ** 2, axis=1)
            ranked_member_indices = member_indices[np.argsort(member_distances)]
            example_ids = [record_features[int(idx)].record_id for idx in ranked_member_indices[:12]]
        stats_mean = cluster_mean_stats(record_features, member_indices)
        summary_clusters.append(
            {
                "cluster_id": cluster_id,
                "style_code": style_codes[cluster_id],
                "style_label": style_slugs[cluster_id],
                "style_prompt": style_prompts[cluster_id],
                "count": int(len(member_indices)),
                "mean_stats": stats_mean,
                "centroid": {
                    "luma_mean": float(centroid[0]),
                    "luma_std": float(centroid[1]),
                    "saturation_mean": float(centroid[2]),
                    "chroma_rg": float(centroid[3]),
                    "chroma_gb": float(centroid[4]),
                    "chroma_rb": float(centroid[5]),
                    "vignette_drop": float(centroid[6]),
                    "core_minus_mid_luma": float(centroid[7]),
                },
                "example_record_ids": example_ids,
            }
        )
    return {
        "num_clusters": len(style_prompts),
        "num_records": len(record_features),
        "clusters": summary_clusters,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster fundus color styles and annotate metadata captions.")
    parser.add_argument("--fit_metadata", type=str, required=True, help="Metadata used to fit color-style clusters.")
    parser.add_argument(
        "--apply_metadata",
        type=str,
        nargs="+",
        required=True,
        help="One or more metadata files to annotate using the fitted clusters.",
    )
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory for relative image paths.")
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--padding_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Parallel workers used to compute fundus color features.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_colorstyle",
        help="Suffix inserted before .jsonl for annotated metadata outputs.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional path for the clustering summary JSON. Defaults near fit_metadata.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    fit_metadata = Path(args.fit_metadata).expanduser().resolve()
    apply_metadata = [Path(path).expanduser().resolve() for path in args.apply_metadata]

    all_records: list[dict] = []
    for metadata_path in [fit_metadata] + apply_metadata:
        all_records.extend(load_jsonl(metadata_path))

    unique_tasks: dict[Path, FeatureTask] = {}
    for record in all_records:
        image_path = resolve_path(root_dir, record.get("image") or record.get("image_path"))
        if image_path not in unique_tasks:
            mask_value = record.get("mask") or record.get("mask_path")
            unique_tasks[image_path] = FeatureTask(
                image_path=image_path,
                mask_path=resolve_path(root_dir, mask_value) if mask_value else None,
                record_id=str(record["id"]),
            )

    feature_cache: dict[Path, RecordFeature] = {}
    ordered_tasks = list(unique_tasks.values())
    num_workers = max(1, int(args.num_workers))

    if num_workers == 1:
        for task in ordered_tasks:
            feature_cache[task.image_path] = compute_record_feature_from_task(
                task,
                image_size=int(args.image_size),
                padding_ratio=float(args.padding_ratio),
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    compute_record_feature_from_task,
                    task,
                    image_size=int(args.image_size),
                    padding_ratio=float(args.padding_ratio),
                ): task.image_path
                for task in ordered_tasks
            }
            total = len(futures)
            completed = 0
            for future in as_completed(futures):
                image_path = futures[future]
                feature_cache[image_path] = future.result()
                completed += 1
                if completed == total or completed % 256 == 0:
                    print(f"[progress] feature stats {completed}/{total}", flush=True)

    fit_records = load_jsonl(fit_metadata)
    fit_features = [
        feature_cache[resolve_path(root_dir, record.get("image") or record.get("image_path"))]
        for record in fit_records
    ]
    fit_matrix_raw = stack_features(fit_features)
    fit_matrix_norm, feature_mean, feature_std = standardize_features(fit_matrix_raw)
    feature_weights = build_feature_weights(fit_matrix_norm.shape[1])
    fit_matrix_weighted = fit_matrix_norm * feature_weights[None, :]
    centroids_weighted, fit_assignments = fit_kmeans(
        fit_matrix_weighted,
        num_clusters=int(args.num_clusters),
        seed=int(args.seed),
    )
    centroids_norm = centroids_weighted / feature_weights[None, :]
    centroids_raw = centroids_norm * feature_std + feature_mean

    cluster_order = sorted(
        range(int(args.num_clusters)),
        key=lambda cluster_id: (
            -float(centroids_raw[cluster_id][0]),
            -float(centroids_raw[cluster_id][5]),
            float(centroids_raw[cluster_id][6]),
        ),
    )
    ordered_centroids_norm = centroids_weighted[cluster_order]
    ordered_centroids_raw = centroids_raw[cluster_order]
    remapped_fit_assignments = np.asarray([cluster_order.index(int(idx)) for idx in fit_assignments], dtype=np.int64)

    style_prompts = []
    style_codes = []
    style_slugs = []
    for new_cluster_id in range(int(args.num_clusters)):
        member_indices = np.where(remapped_fit_assignments == new_cluster_id)[0]
        stats_mean = cluster_mean_stats(fit_features, member_indices)
        style_code = f"style_{new_cluster_id:02d}"
        prompt = make_style_prompt(stats_mean)
        style_codes.append(style_code)
        style_prompts.append(prompt)
        style_slugs.append(f"{style_code}_{make_style_slug(prompt)}")

    summary = build_summary(
        centroids_raw=ordered_centroids_raw,
        centroids_norm=ordered_centroids_norm,
        assignments=remapped_fit_assignments,
        features_norm=fit_matrix_weighted,
        record_features=fit_features,
        style_codes=style_codes,
        style_prompts=style_prompts,
        style_slugs=style_slugs,
    )

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
    else:
        summary_path = fit_metadata.with_name(fit_metadata.stem + str(args.output_suffix) + "_clusters.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for metadata_path in apply_metadata:
        records = load_jsonl(metadata_path)
        enriched_records = []
        features = [
            feature_cache[resolve_path(root_dir, record.get("image") or record.get("image_path"))]
            for record in records
        ]
        matrix_raw = stack_features(features)
        matrix_norm = (matrix_raw - feature_mean) / feature_std
        matrix_weighted = matrix_norm * feature_weights[None, :]
        assignments = assign_clusters(matrix_weighted, ordered_centroids_norm)
        for record, cluster_id in zip(records, assignments.tolist()):
            enriched_records.append(
                enrich_record(
                    record,
                    cluster_id=int(cluster_id),
                    style_code=style_codes[int(cluster_id)],
                    style_prompt=style_prompts[int(cluster_id)],
                    style_slug=style_slugs[int(cluster_id)],
                )
            )
        output_path = metadata_path.with_name(metadata_path.stem + str(args.output_suffix) + metadata_path.suffix)
        write_jsonl(output_path, enriched_records)
        print(f"[ok] wrote {output_path} ({len(enriched_records)} records)", flush=True)

    print(f"[ok] wrote cluster summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
