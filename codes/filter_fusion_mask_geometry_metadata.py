from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from struct_mask_utils import FUSION_MASK_CHANNELS


OPTIC_DISC_COLOR = next(color for name, color in FUSION_MASK_CHANNELS if name == "optic_disc")
VESSEL_COLOR = next(color for name, color in FUSION_MASK_CHANNELS if name == "vessels")
KNOWN_COLORS = {tuple(color) for _, color in FUSION_MASK_CHANNELS}


def get_available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def infer_expected_disc_side(record_id: str) -> str | None:
    stem = Path(record_id).stem.lower()
    if stem.endswith("_left"):
        return "left"
    if stem.endswith("_right"):
        return "right"
    return None


def quantile_or_default(values: np.ndarray, q: float, default: float) -> float:
    if values.size == 0:
        return default
    return float(np.quantile(values, q))


def compute_mask_metrics(mask_path: Path, record_id: str) -> dict[str, float | int | bool | str | None]:
    mask_image = Image.open(mask_path).convert("RGB")
    arr = np.asarray(mask_image, dtype=np.uint8)
    height, width = arr.shape[:2]
    square = bool(width == height)

    flat_colors = np.unique(arr.reshape(-1, 3), axis=0)
    unknown_colors = sorted(
        tuple(int(v) for v in color)
        for color in flat_colors.tolist()
        if tuple(int(v) for v in color) != (0, 0, 0) and tuple(int(v) for v in color) not in KNOWN_COLORS
    )

    nonzero = np.any(arr != 0, axis=-1)
    vessel = np.all(arr == np.asarray(VESSEL_COLOR, dtype=np.uint8), axis=-1)
    disc = np.all(arr == np.asarray(OPTIC_DISC_COLOR, dtype=np.uint8), axis=-1)

    metrics: dict[str, float | int | bool | str | None] = {
        "mask_width": int(width),
        "mask_height": int(height),
        "mask_square": square,
        "unknown_palette_colors": len(unknown_colors),
        "unknown_palette_examples": unknown_colors[:8],
        "nonzero_ratio": float(nonzero.mean()),
        "nonzero_outside_circle_ratio": 0.0,
        "vessel_pixel_count": int(vessel.sum()),
        "vessel_span_x": 0.0,
        "vessel_span_y": 0.0,
        "optic_disc_present": bool(disc.any()),
        "optic_disc_pixel_count": int(disc.sum()),
        "optic_disc_area_ratio": float(disc.mean()),
        "optic_disc_x_norm": None,
        "optic_disc_y_norm": None,
        "optic_disc_edge_margin": None,
        "optic_disc_side": None,
        "laterality_expected_side": infer_expected_disc_side(record_id),
        "laterality_match": None,
    }

    if nonzero.any():
        yy, xx = np.indices((height, width), dtype=np.float32)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        radius = min(width, height) / 2.0
        inside_circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (radius**2)
        outside_ratio = float((nonzero & ~inside_circle).sum() / max(int(nonzero.sum()), 1))
        metrics["nonzero_outside_circle_ratio"] = outside_ratio

    if vessel.any():
        ys, xs = np.nonzero(vessel)
        metrics["vessel_span_x"] = float((xs.max() - xs.min() + 1) / max(width, 1))
        metrics["vessel_span_y"] = float((ys.max() - ys.min() + 1) / max(height, 1))

    if disc.any():
        ys, xs = np.nonzero(disc)
        x_norm = float(xs.mean()) / max(float(width - 1), 1.0)
        y_norm = float(ys.mean()) / max(float(height - 1), 1.0)
        edge_margin = float(min(x_norm, 1.0 - x_norm, y_norm, 1.0 - y_norm))
        side = "left" if x_norm < 0.5 else "right"
        expected_side = metrics["laterality_expected_side"]
        laterality_match = None if expected_side is None else (side == expected_side)
        metrics.update(
            {
                "optic_disc_x_norm": x_norm,
                "optic_disc_y_norm": y_norm,
                "optic_disc_edge_margin": edge_margin,
                "optic_disc_side": side,
                "laterality_match": laterality_match,
            }
        )

    return metrics


def classify(metrics: dict[str, float | int | bool | str | None], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []

    if not bool(metrics["mask_square"]):
        reasons.append("mask_not_square")

    if int(metrics["unknown_palette_colors"]) > 0:
        reasons.append("unknown_palette_color")

    if float(metrics["nonzero_outside_circle_ratio"]) > args.max_nonzero_outside_circle_ratio:
        reasons.append("mask_outside_canonical_circle")

    if int(metrics["vessel_pixel_count"]) < args.min_vessel_pixel_count:
        reasons.append("vessel_too_sparse")
    if float(metrics["vessel_span_x"]) < args.min_vessel_span_x:
        reasons.append("vessel_span_x_too_narrow")
    if float(metrics["vessel_span_y"]) < args.min_vessel_span_y:
        reasons.append("vessel_span_y_too_narrow")

    if args.require_optic_disc and not bool(metrics["optic_disc_present"]):
        reasons.append("optic_disc_missing")
        return reasons

    if bool(metrics["optic_disc_present"]):
        disc_area_ratio = float(metrics["optic_disc_area_ratio"])
        edge_margin = float(metrics["optic_disc_edge_margin"] or 0.0)
        if disc_area_ratio < args.min_optic_disc_area_ratio:
            reasons.append("optic_disc_too_small")
        if disc_area_ratio > args.max_optic_disc_area_ratio:
            reasons.append("optic_disc_too_large")
        if edge_margin < args.min_optic_disc_edge_margin:
            reasons.append("optic_disc_too_close_to_edge")
        if args.enforce_laterality and metrics["laterality_match"] is False:
            reasons.append("optic_disc_laterality_mismatch")

    return reasons


def process_record(payload: tuple[dict, str, dict[str, float | int | bool]]) -> tuple[dict, dict, list[str]]:
    record, root_dir_str, settings = payload
    root_dir = Path(root_dir_str)
    mask_path = Path(record["mask"])
    mask_path = mask_path if mask_path.is_absolute() else (root_dir / mask_path)

    metrics = compute_mask_metrics(mask_path, str(record["id"]))

    reasons = []
    if not bool(metrics["mask_square"]):
        reasons.append("mask_not_square")
    if int(metrics["unknown_palette_colors"]) > 0:
        reasons.append("unknown_palette_color")
    if float(metrics["nonzero_outside_circle_ratio"]) > float(settings["max_nonzero_outside_circle_ratio"]):
        reasons.append("mask_outside_canonical_circle")
    if int(metrics["vessel_pixel_count"]) < int(settings["min_vessel_pixel_count"]):
        reasons.append("vessel_too_sparse")
    if float(metrics["vessel_span_x"]) < float(settings["min_vessel_span_x"]):
        reasons.append("vessel_span_x_too_narrow")
    if float(metrics["vessel_span_y"]) < float(settings["min_vessel_span_y"]):
        reasons.append("vessel_span_y_too_narrow")

    require_optic_disc = bool(settings["require_optic_disc"])
    if require_optic_disc and not bool(metrics["optic_disc_present"]):
        reasons.append("optic_disc_missing")
        return record, metrics, reasons

    if bool(metrics["optic_disc_present"]):
        disc_area_ratio = float(metrics["optic_disc_area_ratio"])
        edge_margin = float(metrics["optic_disc_edge_margin"] or 0.0)
        if disc_area_ratio < float(settings["min_optic_disc_area_ratio"]):
            reasons.append("optic_disc_too_small")
        if disc_area_ratio > float(settings["max_optic_disc_area_ratio"]):
            reasons.append("optic_disc_too_large")
        if edge_margin < float(settings["min_optic_disc_edge_margin"]):
            reasons.append("optic_disc_too_close_to_edge")
        if bool(settings["enforce_laterality"]) and metrics["laterality_match"] is False:
            reasons.append("optic_disc_laterality_mismatch")

    return record, metrics, reasons


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter fusion masks with geometry and optic-disc QA rules.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_metadata", required=True)
    parser.add_argument("--excluded_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--jobs", type=int, default=min(8, get_available_cpu_count()))
    parser.add_argument("--require_optic_disc", type=int, default=1)
    parser.add_argument("--enforce_laterality", type=int, default=1)
    parser.add_argument("--min_optic_disc_area_ratio", type=float, default=0.008)
    parser.add_argument("--max_optic_disc_area_ratio", type=float, default=0.035)
    parser.add_argument("--min_optic_disc_edge_margin", type=float, default=0.12)
    parser.add_argument("--min_vessel_pixel_count", type=int, default=1200)
    parser.add_argument("--min_vessel_span_x", type=float, default=0.82)
    parser.add_argument("--min_vessel_span_y", type=float, default=0.82)
    parser.add_argument("--max_nonzero_outside_circle_ratio", type=float, default=0.12)
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    root_dir = Path(args.root_dir).resolve()
    output_metadata = Path(args.output_metadata).resolve()
    excluded_jsonl = Path(args.excluded_jsonl).resolve()
    report_json = Path(args.report_json).resolve()

    records = []
    with metadata_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    settings = {
        "require_optic_disc": bool(args.require_optic_disc),
        "enforce_laterality": bool(args.enforce_laterality),
        "min_optic_disc_area_ratio": args.min_optic_disc_area_ratio,
        "max_optic_disc_area_ratio": args.max_optic_disc_area_ratio,
        "min_optic_disc_edge_margin": args.min_optic_disc_edge_margin,
        "min_vessel_pixel_count": args.min_vessel_pixel_count,
        "min_vessel_span_x": args.min_vessel_span_x,
        "min_vessel_span_y": args.min_vessel_span_y,
        "max_nonzero_outside_circle_ratio": args.max_nonzero_outside_circle_ratio,
    }

    payloads = [(record, str(root_dir), settings) for record in records]
    iterator = None
    pool = None
    if args.jobs <= 1:
        iterator = map(process_record, payloads)
    else:
        pool = Pool(processes=args.jobs)
        iterator = pool.imap_unordered(process_record, payloads, chunksize=32)

    kept: list[dict] = []
    excluded: list[dict] = []
    reason_counts: Counter[str] = Counter()
    reason_examples: defaultdict[str, list[str]] = defaultdict(list)
    disc_area_values: list[float] = []
    disc_margin_values: list[float] = []
    vessel_span_x_values: list[float] = []
    vessel_span_y_values: list[float] = []

    for idx, (record, metrics, reasons) in enumerate(iterator, start=1):
        if bool(metrics["optic_disc_present"]):
            disc_area_values.append(float(metrics["optic_disc_area_ratio"]))
            disc_margin_values.append(float(metrics["optic_disc_edge_margin"] or 0.0))
        vessel_span_x_values.append(float(metrics["vessel_span_x"]))
        vessel_span_y_values.append(float(metrics["vessel_span_y"]))

        if reasons:
            row = dict(record)
            row["exclude_reasons"] = reasons
            row["mask_qc_metrics"] = metrics
            excluded.append(row)
            for reason in reasons:
                reason_counts[reason] += 1
                if len(reason_examples[reason]) < 12:
                    reason_examples[reason].append(str(record["id"]))
        else:
            kept.append(record)

        if idx % 500 == 0:
            print(f"[{idx}/{len(records)}] mask QC filtered", flush=True)

    if pool is not None:
        pool.close()
        pool.join()

    kept = sorted(kept, key=lambda row: str(row["id"]))
    excluded = sorted(excluded, key=lambda row: str(row["id"]))

    write_jsonl(output_metadata, kept)
    write_jsonl(excluded_jsonl, excluded)

    report = {
        "input_metadata": str(metadata_path),
        "output_metadata": str(output_metadata),
        "excluded_jsonl": str(excluded_jsonl),
        "input_records": len(records),
        "kept_records": len(kept),
        "excluded_records": len(excluded),
        "reason_counts": dict(reason_counts),
        "reason_examples": dict(reason_examples),
        "thresholds": {
            "require_optic_disc": bool(args.require_optic_disc),
            "enforce_laterality": bool(args.enforce_laterality),
            "min_optic_disc_area_ratio": args.min_optic_disc_area_ratio,
            "max_optic_disc_area_ratio": args.max_optic_disc_area_ratio,
            "min_optic_disc_edge_margin": args.min_optic_disc_edge_margin,
            "min_vessel_pixel_count": args.min_vessel_pixel_count,
            "min_vessel_span_x": args.min_vessel_span_x,
            "min_vessel_span_y": args.min_vessel_span_y,
            "max_nonzero_outside_circle_ratio": args.max_nonzero_outside_circle_ratio,
            "jobs": args.jobs,
        },
        "metric_quantiles": {
            "optic_disc_area_ratio": {
                "p01": quantile_or_default(np.asarray(disc_area_values, dtype=np.float32), 0.01, 0.0),
                "p05": quantile_or_default(np.asarray(disc_area_values, dtype=np.float32), 0.05, 0.0),
                "p50": quantile_or_default(np.asarray(disc_area_values, dtype=np.float32), 0.50, 0.0),
                "p95": quantile_or_default(np.asarray(disc_area_values, dtype=np.float32), 0.95, 0.0),
                "p99": quantile_or_default(np.asarray(disc_area_values, dtype=np.float32), 0.99, 0.0),
            },
            "optic_disc_edge_margin": {
                "p01": quantile_or_default(np.asarray(disc_margin_values, dtype=np.float32), 0.01, 0.0),
                "p05": quantile_or_default(np.asarray(disc_margin_values, dtype=np.float32), 0.05, 0.0),
                "p50": quantile_or_default(np.asarray(disc_margin_values, dtype=np.float32), 0.50, 0.0),
                "p95": quantile_or_default(np.asarray(disc_margin_values, dtype=np.float32), 0.95, 0.0),
                "p99": quantile_or_default(np.asarray(disc_margin_values, dtype=np.float32), 0.99, 0.0),
            },
            "vessel_span_x": {
                "p01": quantile_or_default(np.asarray(vessel_span_x_values, dtype=np.float32), 0.01, 0.0),
                "p05": quantile_or_default(np.asarray(vessel_span_x_values, dtype=np.float32), 0.05, 0.0),
                "p50": quantile_or_default(np.asarray(vessel_span_x_values, dtype=np.float32), 0.50, 0.0),
                "p95": quantile_or_default(np.asarray(vessel_span_x_values, dtype=np.float32), 0.95, 0.0),
                "p99": quantile_or_default(np.asarray(vessel_span_x_values, dtype=np.float32), 0.99, 0.0),
            },
            "vessel_span_y": {
                "p01": quantile_or_default(np.asarray(vessel_span_y_values, dtype=np.float32), 0.01, 0.0),
                "p05": quantile_or_default(np.asarray(vessel_span_y_values, dtype=np.float32), 0.05, 0.0),
                "p50": quantile_or_default(np.asarray(vessel_span_y_values, dtype=np.float32), 0.50, 0.0),
                "p95": quantile_or_default(np.asarray(vessel_span_y_values, dtype=np.float32), 0.95, 0.0),
                "p99": quantile_or_default(np.asarray(vessel_span_y_values, dtype=np.float32), 0.99, 0.0),
            },
        },
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
