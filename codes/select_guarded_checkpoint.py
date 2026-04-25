import argparse
import json
from pathlib import Path


DEFAULT_GUARDS = (
    ("global_ssim", "higher", "structure"),
    ("mask_ssim", "higher", "structure"),
    ("global_l1", "lower", "structure"),
    ("mask_l1", "lower", "structure"),
    ("fundus_mean_abs_error", "lower", "color"),
    ("fundus_chroma_l1", "lower", "color"),
    ("fundus_rb_gap_abs_error", "lower", "color"),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_mean(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, dict) or value.get("mean") is None:
        raise KeyError(f"Metric {name!r} has no mean value")
    return float(value["mean"])


def compare(candidate: float, baseline: float, direction: str, epsilon: float) -> tuple[bool, float]:
    denom = max(abs(baseline), 1e-8)
    if direction == "higher":
        delta = candidate - baseline
        return candidate + epsilon >= baseline, delta / denom
    if direction == "lower":
        delta = baseline - candidate
        return candidate <= baseline + epsilon, delta / denom
    raise ValueError(f"Unsupported direction: {direction}")


def iter_metric_files(eval_root: Path) -> list[Path]:
    return sorted(
        path
        for path in eval_root.glob("*/structural_metrics.json")
        if path.parent.name.isdigit()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select only checkpoints that do not regress against a baseline on "
            "structure and color metrics."
        )
    )
    parser.add_argument("--baseline_json", required=True, help="Baseline structural_metrics.json, usually the good 1k eval.")
    parser.add_argument("--eval_root", required=True, help="Run checkpoints/eval directory containing step metrics.")
    parser.add_argument("--output_json", required=True, help="Where to write guarded selection summary.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Numerical tolerance for no-regression checks.")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json).expanduser().resolve()
    eval_root = Path(args.eval_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    baseline = load_json(baseline_path)
    metric_files = iter_metric_files(eval_root)
    if not metric_files:
        raise FileNotFoundError(f"No step structural_metrics.json files found under {eval_root}")

    candidates = []
    accepted = []
    for metric_file in metric_files:
        metrics = load_json(metric_file)
        checks = []
        score = 0.0
        passed = True
        for name, direction, group in DEFAULT_GUARDS:
            base_value = metric_mean(baseline, name)
            cand_value = metric_mean(metrics, name)
            ok, normalized_delta = compare(cand_value, base_value, direction, args.epsilon)
            passed = passed and ok
            score += normalized_delta
            checks.append(
                {
                    "metric": name,
                    "group": group,
                    "direction": direction,
                    "baseline": base_value,
                    "candidate": cand_value,
                    "normalized_delta": normalized_delta,
                    "passed": ok,
                }
            )

        item = {
            "step": metric_file.parent.name,
            "metrics_json": str(metric_file),
            "eval_dir": str(metric_file.parent),
            "generated_dir": str(metric_file.parent / "generated"),
            "score": score,
            "passed": passed,
            "checks": checks,
        }
        candidates.append(item)
        if passed:
            accepted.append(item)

    best = max(accepted, key=lambda item: (item["score"], item["step"])) if accepted else None
    result = {
        "baseline_json": str(baseline_path),
        "eval_root": str(eval_root),
        "guard_metrics": [
            {"metric": name, "direction": direction, "group": group}
            for name, direction, group in DEFAULT_GUARDS
        ],
        "accepted_count": len(accepted),
        "best": best,
        "candidates": candidates,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if best is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
