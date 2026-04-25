import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split metadata_all into train_full and example holdout.")
    parser.add_argument("--metadata_all", required=True, type=str)
    parser.add_argument("--metadata_train_full", required=True, type=str)
    parser.add_argument("--metadata_example", required=True, type=str)
    parser.add_argument("--build_report", required=True, type=str)
    parser.add_argument("--example_count", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    metadata_all = Path(args.metadata_all).expanduser().resolve()
    metadata_train_full = Path(args.metadata_train_full).expanduser().resolve()
    metadata_example = Path(args.metadata_example).expanduser().resolve()
    build_report_path = Path(args.build_report).expanduser().resolve()

    if args.example_count < 0:
        raise ValueError("example_count must be >= 0")
    if not metadata_all.exists():
        raise FileNotFoundError(f"metadata_all not found: {metadata_all}")

    records = load_jsonl(metadata_all)
    if not records:
        raise RuntimeError("metadata_all is empty")

    records = sorted(records, key=lambda x: str(x["id"]))
    shuffled = records[:]
    random.Random(args.seed).shuffle(shuffled)

    example_size = min(args.example_count, len(shuffled))
    example_records = sorted(shuffled[:example_size], key=lambda x: str(x["id"]))
    train_full_records = sorted(shuffled[example_size:], key=lambda x: str(x["id"]))

    save_jsonl(metadata_train_full, train_full_records)
    save_jsonl(metadata_example, example_records)

    report = {}
    if build_report_path.exists():
        report = json.loads(build_report_path.read_text(encoding="utf-8"))
    report["seed"] = args.seed
    report["example_count"] = args.example_count
    report["matched_records"] = len(records)
    report["train_full_records"] = len(train_full_records)
    report["example_records"] = len(example_records)
    report["val_ratio"] = 0.0
    report["val_records"] = 0
    build_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"metadata_all={len(records)}")
    print(f"train_full={len(train_full_records)}")
    print(f"example={len(example_records)}")
    print(f"report={build_report_path}")


if __name__ == "__main__":
    main()
