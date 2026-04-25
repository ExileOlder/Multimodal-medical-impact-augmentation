from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PARSER_STEP_RE = re.compile(r"step=(\d{5})/(\d+)\s+loss=([0-9.]+)")
PARSER_VAL_RE = re.compile(
    r"\[val\]\s+step=(\d+)\s+loss=([0-9.]+).*?weighted_lesion_dice=([0-9.]+).*?selection_score=([0-9.]+)"
)
STAGE_STEP_RE = re.compile(r"step=(\d+)/(\d+),\s+loss=([0-9.]+)")
STAGE_SUMMARY_RE = re.compile(r"\(step=(\d{7})\)\s+Train Loss:\s+([0-9.]+)")


def parse_parser_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    train_match = None
    for match in PARSER_STEP_RE.finditer(text):
        train_match = match

    val_match = None
    for match in PARSER_VAL_RE.finditer(text):
        val_match = match

    return {
        "path": str(path),
        "exists": path.exists(),
        "train_step": int(train_match.group(1)) if train_match else None,
        "train_step_total": int(train_match.group(2)) if train_match else None,
        "train_loss": float(train_match.group(3)) if train_match else None,
        "val_step": int(val_match.group(1)) if val_match else None,
        "val_loss": float(val_match.group(2)) if val_match else None,
        "weighted_lesion_dice": float(val_match.group(3)) if val_match else None,
        "selection_score": float(val_match.group(4)) if val_match else None,
    }


def parse_stage_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    step_match = None
    for match in STAGE_STEP_RE.finditer(text):
        step_match = match

    summary_match = None
    for match in STAGE_SUMMARY_RE.finditer(text):
        summary_match = match

    return {
        "path": str(path),
        "exists": path.exists(),
        "train_step": int(step_match.group(1)) if step_match else None,
        "train_step_total": int(step_match.group(2)) if step_match else None,
        "latest_inline_loss": float(step_match.group(3)) if step_match else None,
        "latest_summary_step": int(summary_match.group(1)) if summary_match else None,
        "latest_summary_loss": float(summary_match.group(2)) if summary_match else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize old-vs-new scheme training progress.")
    parser.add_argument(
        "--parser-log",
        type=str,
        default="./results/setup/4090/train_fusion_parser_10k_parallel.log",
    )
    parser.add_argument(
        "--stagea-log",
        type=str,
        default="./results/setup/4090/train_mainline_stageA_10k_parallel.log",
    )
    parser.add_argument(
        "--stageb-log",
        type=str,
        default="./results/setup/4090/train_stageb_lesioncrop.log",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./results/setup/4090/scheme_progress_summary.json",
    )
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    parser_log = (code_root / args.parser_log).resolve() if not Path(args.parser_log).is_absolute() else Path(args.parser_log)
    stagea_log = (code_root / args.stagea_log).resolve() if not Path(args.stagea_log).is_absolute() else Path(args.stagea_log)
    stageb_log = (code_root / args.stageb_log).resolve() if not Path(args.stageb_log).is_absolute() else Path(args.stageb_log)
    out_path = (code_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    summary = {
        "old_scheme_parser": parse_parser_log(parser_log),
        "new_scheme_stageA": parse_stage_log(stagea_log),
        "new_scheme_stageB": parse_stage_log(stageb_log),
        "notes": {
            "old_scheme": "parser-centered route",
            "new_scheme": "stageA global base-image -> stageB local lesion editing -> stageC consistency screening",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
