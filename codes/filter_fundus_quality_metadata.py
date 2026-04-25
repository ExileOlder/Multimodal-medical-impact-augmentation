import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFilter


def get_available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def get_fundus_mask(arr: np.ndarray, threshold: float = 12.0) -> np.ndarray:
    return arr.mean(axis=2) > threshold


def get_radius_map(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return np.ones(mask.shape, dtype=np.float32)
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    max_r = float(np.max(np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2))) + 1e-6
    yy, xx = np.indices(mask.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max_r
    return rr.astype(np.float32)


def compute_quality_metrics(image_path: Path, resize_max: int = 1024) -> dict:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > resize_max:
        scale = resize_max / float(max(img.size))
        img = img.resize(
            (max(32, int(img.size[0] * scale)), max(32, int(img.size[1] * scale))),
            Image.Resampling.BILINEAR,
        )

    arr = np.asarray(img).astype(np.float32)
    mask = get_fundus_mask(arr)
    if mask.sum() < 256:
        return {
            "valid": False,
            "edge_green": 0.0,
            "edge_blue": 0.0,
            "edge_luma_ratio": 1.0,
            "p995_luma": 0.0,
            "noise": 0.0,
        }

    rgb = arr / 255.0
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    radius = get_radius_map(mask)
    ring = mask & (radius >= 0.84)
    core = mask & (radius <= 0.55)

    if ring.sum() < 64 or core.sum() < 64:
        return {
            "valid": False,
            "edge_green": 0.0,
            "edge_blue": 0.0,
            "edge_luma_ratio": 1.0,
            "p995_luma": float(np.quantile(luma[mask], 0.995)),
            "noise": 0.0,
        }

    edge_green = float(rgb[..., 1][ring].mean() - rgb[..., 1][core].mean())
    edge_blue = float(rgb[..., 2][ring].mean() - rgb[..., 2][core].mean())
    edge_luma_ratio = float(luma[ring].mean() / max(luma[core].mean(), 1e-6))
    p995_luma = float(np.quantile(luma[mask], 0.995))

    den = np.asarray(img.filter(ImageFilter.MedianFilter(size=3))).astype(np.float32) / 255.0
    noise_map = np.mean(np.abs(rgb - den), axis=2)
    noise = float(noise_map[mask].mean())

    return {
        "valid": True,
        "edge_green": edge_green,
        "edge_blue": edge_blue,
        "edge_luma_ratio": edge_luma_ratio,
        "p995_luma": p995_luma,
        "noise": noise,
    }


def classify(metrics: dict, args: argparse.Namespace) -> list[str]:
    if not metrics["valid"]:
        return ["invalid_fundus_mask"]
    reasons = []
    if metrics["edge_green"] > args.max_edge_green:
        reasons.append("edge_green_flare")
    if metrics["edge_blue"] > args.max_edge_blue:
        reasons.append("edge_blue_flare")
    if metrics["edge_luma_ratio"] > args.max_edge_luma_ratio:
        reasons.append("edge_luma_flare")
    if metrics["p995_luma"] > args.max_p995_luma:
        reasons.append("overbright")
    if metrics["noise"] > args.max_noise:
        reasons.append("high_noise")
    return reasons


def process_record(payload: tuple[dict, str, int, dict]) -> tuple[dict, dict, list[str]]:
    record, root_dir_str, resize_max, thresholds = payload
    root_dir = Path(root_dir_str)
    image_path = Path(record["image"])
    image_path = image_path if image_path.is_absolute() else (root_dir / image_path)
    metrics = compute_quality_metrics(image_path, resize_max=resize_max)

    reasons = []
    if not metrics["valid"]:
        reasons = ["invalid_fundus_mask"]
    else:
        if metrics["edge_green"] > thresholds["max_edge_green"]:
            reasons.append("edge_green_flare")
        if metrics["edge_blue"] > thresholds["max_edge_blue"]:
            reasons.append("edge_blue_flare")
        if metrics["edge_luma_ratio"] > thresholds["max_edge_luma_ratio"]:
            reasons.append("edge_luma_flare")
        if metrics["p995_luma"] > thresholds["max_p995_luma"]:
            reasons.append("overbright")
        if metrics["noise"] > thresholds["max_noise"]:
            reasons.append("high_noise")
    return record, metrics, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter severe fundus quality outliers from metadata.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_metadata", required=True)
    parser.add_argument("--excluded_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--resize_max", type=int, default=768)
    parser.add_argument("--max_edge_green", type=float, default=0.08)
    parser.add_argument("--max_edge_blue", type=float, default=0.05)
    parser.add_argument("--max_edge_luma_ratio", type=float, default=1.12)
    parser.add_argument("--max_p995_luma", type=float, default=0.86)
    parser.add_argument("--max_noise", type=float, default=0.0045)
    parser.add_argument("--jobs", type=int, default=min(8, get_available_cpu_count()))
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    records = []
    with open(args.metadata, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    kept = []
    excluded = []
    counters = {}
    thresholds = {
        "max_edge_green": args.max_edge_green,
        "max_edge_blue": args.max_edge_blue,
        "max_edge_luma_ratio": args.max_edge_luma_ratio,
        "max_p995_luma": args.max_p995_luma,
        "max_noise": args.max_noise,
    }
    payloads = [
        (record, str(root_dir), args.resize_max, thresholds)
        for record in records
    ]

    if args.jobs <= 1:
        iterator = map(process_record, payloads)
    else:
        pool = Pool(processes=args.jobs)
        iterator = pool.imap_unordered(process_record, payloads, chunksize=32)

    for idx, (record, metrics, reasons) in enumerate(iterator, start=1):
        if reasons:
            excluded_record = dict(record)
            excluded_record["exclude_reasons"] = reasons
            excluded_record["quality_metrics"] = metrics
            excluded.append(excluded_record)
            for reason in reasons:
                counters[reason] = counters.get(reason, 0) + 1
        else:
            kept.append(record)
        if idx % 500 == 0:
            print(f"[{idx}/{len(records)}] quality filtered", flush=True)

    if args.jobs > 1:
        pool.close()
        pool.join()

    output_metadata = Path(args.output_metadata).resolve()
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metadata, "w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    excluded_jsonl = Path(args.excluded_jsonl).resolve()
    with open(excluded_jsonl, "w", encoding="utf-8") as handle:
        for record in excluded:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "input_metadata": str(Path(args.metadata).resolve()),
        "output_metadata": str(output_metadata),
        "excluded_jsonl": str(excluded_jsonl),
        "input_records": len(records),
        "kept_records": len(kept),
        "excluded_records": len(excluded),
        "reason_counts": counters,
        "thresholds": {
            "max_edge_green": args.max_edge_green,
            "max_edge_blue": args.max_edge_blue,
            "max_edge_luma_ratio": args.max_edge_luma_ratio,
            "max_p995_luma": args.max_p995_luma,
            "max_noise": args.max_noise,
            "resize_max": args.resize_max,
            "jobs": args.jobs,
        },
    }
    with open(args.report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
