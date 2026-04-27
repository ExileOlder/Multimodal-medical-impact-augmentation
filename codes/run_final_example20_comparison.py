import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from fundus_geometry import canonicalize_fundus_image


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root_dir / path).resolve()


def prepare_real_dir(records: list[dict], root_dir: Path, real_dir: Path, image_size: int) -> None:
    real_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        record_id = str(record["id"])
        out_path = real_dir / f"{record_id}.png"
        if out_path.exists():
            continue
        image_path = resolve_path(root_dir, record["image"])
        with Image.open(image_path).convert("RGB") as image:
            image = canonicalize_fundus_image(image, output_size=image_size, padding_ratio=0.01)
            image.save(out_path)


def run_inference(
    record: dict,
    root_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    mode: str,
) -> None:
    record_id = str(record["id"])
    out_path = out_dir / f"{record_id}.png"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {mode} {record_id}: exists")
        return

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "inference_mask.py"),
        "--model",
        args.model,
        "--base_ckpt",
        str(args.base_ckpt),
        "--prompt",
        str(record["caption"]),
        "--out_dir",
        str(out_dir),
        "--output_name",
        record_id,
        "--seed",
        str(args.seed),
        "--cfg_scale",
        str(args.cfg_scale),
        "--num_sampling_steps",
        str(args.num_sampling_steps),
        "--sampling_method",
        args.sampling_method,
        "--precision",
        args.precision,
        "--vae",
        args.vae,
        "--image_size",
        str(args.image_size),
        "--struct_mask_channels",
        str(args.struct_mask_channels),
        "--mask_scale",
        str(args.mask_scale),
        "--tokenizer_path",
        str(args.tokenizer_path),
        "--local_diffusers_model_root",
        str(args.local_diffusers_model_root),
        "--qk_norm",
    ]
    if mode == "ours":
        cmd.extend(["--adapter_ckpt", str(args.adapter_ckpt)])
        cmd.extend(["--mask_path", str(resolve_path(root_dir, record["mask"]))])
    elif mode == "base_only":
        cmd.append("--disable_mask_condition")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"[run] {mode} {record_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(cmd, check=True, env=env)


def run_structural_eval(metadata: Path, root_dir: Path, generated_dir: Path, output_json: Path) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_structural_metrics.py"),
        "--metadata",
        str(metadata),
        "--generated_dir",
        str(generated_dir),
        "--root_dir",
        str(root_dir),
        "--output_json",
        str(output_json),
        "--topk",
        "20",
        "--alignment_mode",
        "canonical_square",
    ]
    subprocess.run(cmd, check=True)


def run_fid_kid(real_dir: Path, fake_dir: Path, output_json: Path) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "compute_fid_kid.py"),
        "--real_dir",
        str(real_dir),
        "--fake_dir",
        str(fake_dir),
        "--output_json",
        str(output_json),
        "--batch_size",
        "8",
        "--num_subsets",
        "100",
        "--max_subset_size",
        "20",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    code_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate and evaluate final examples_20 base-only vs ours.")
    parser.add_argument("--metadata", type=Path, default=code_root / "data/final/examples_20/metadata_examples_20_from_final_compact.jsonl")
    parser.add_argument("--root_dir", type=Path, default=code_root / "data")
    parser.add_argument("--base_ckpt", type=Path, default=code_root.parent / "checkpoints")
    parser.add_argument("--adapter_ckpt", type=Path, default=code_root.parent / "checkpoints/stageA_1k_final/adapter.pth")
    parser.add_argument("--out_root", type=Path, default=code_root / "results/final_example20_comparison")
    parser.add_argument("--tokenizer_path", type=Path, default=code_root / "google_gemma-2b")
    parser.add_argument("--local_diffusers_model_root", type=Path, default=code_root / "sdxl-vae")
    parser.add_argument("--model", type=str, default="NextDiT_2B_GQA_patch2")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_sampling_steps", type=int, default=80)
    parser.add_argument("--sampling_method", type=str, default="euler")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--mask_scale", type=float, default=1.0)
    parser.add_argument("--vae", type=str, default="sdxl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--struct_mask_channels", type=int, default=6)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--mode", choices=["all", "base_only", "ours"], default="all")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.metadata = args.metadata.expanduser().resolve()
    args.root_dir = args.root_dir.expanduser().resolve()
    args.base_ckpt = args.base_ckpt.expanduser().resolve()
    args.adapter_ckpt = args.adapter_ckpt.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    args.tokenizer_path = args.tokenizer_path.expanduser().resolve()
    args.local_diffusers_model_root = args.local_diffusers_model_root.expanduser().resolve()

    records = load_jsonl(args.metadata)[: args.limit]
    base_dir = args.out_root / "base_only"
    ours_dir = args.out_root / "ours"
    real_dir = args.out_root / "real_canonical"
    metrics_dir = args.out_root / "metrics"
    prepare_real_dir(records, args.root_dir, real_dir, args.image_size)

    modes = ["base_only", "ours"] if args.mode == "all" else [args.mode]
    if not args.skip_generation:
        for mode in modes:
            out_dir = base_dir if mode == "base_only" else ours_dir
            for record in records:
                run_inference(record, args.root_dir, out_dir, args, mode=mode)

    if not args.skip_metrics:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for mode in modes:
            generated_dir = base_dir if mode == "base_only" else ours_dir
            run_structural_eval(args.metadata, args.root_dir, generated_dir, metrics_dir / f"{mode}_structural_metrics.json")
            run_fid_kid(real_dir, generated_dir, metrics_dir / f"{mode}_fid_kid.json")
    print(f"[ok] comparison outputs saved under {args.out_root}")


if __name__ == "__main__":
    main()
