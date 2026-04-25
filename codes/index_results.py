#!/usr/bin/env python3
"""Build a compact inventory for the results directory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ResultEntry:
    kind: str
    path: str
    size_bytes: int
    files: int
    dirs: int


def summarize_dir(path: Path, kind: str) -> ResultEntry:
    size_bytes = 0
    files = 0
    dirs = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            try:
                size_bytes += item.stat().st_size
            except OSError:
                pass
        elif item.is_dir():
            dirs += 1
    return ResultEntry(kind=kind, path=str(path), size_bytes=size_bytes, files=files, dirs=dirs)


def main() -> int:
    codes_dir = Path(__file__).resolve().parent
    results_dir = codes_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    legacy_names = {
        "inference_results": "legacy_infer",
        "smoke_test": "legacy_smoke",
        "prompt_compare": "legacy_prompt_compare",
        "diag_vae_ckpt": "legacy_diag_vae",
        "diabetic_merged_4090_smoke": "legacy_placeholder",
        "setup_4090": "legacy_setup",
    }

    active_names = {
        "preflight": "preflight",
        "setup": "setup",
        "train": "train",
        "infer": "infer",
    }

    entries: list[ResultEntry] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in active_names:
            kind = active_names[child.name]
        else:
            kind = legacy_names.get(child.name, "other")
        entries.append(summarize_dir(child, kind))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_dir),
        "entries": [asdict(entry) for entry in entries],
    }

    out_path = results_dir / "results_index.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
