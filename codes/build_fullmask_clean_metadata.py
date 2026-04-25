import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_id_set(path: Path) -> set[str]:
    return {record["id"] for record in load_jsonl(path)}


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean full-mask metadata file from allowed-id sets.")
    parser.add_argument("--full_metadata", required=True)
    parser.add_argument("--primary_clean_ids", required=True)
    parser.add_argument("--extra_clean_ids", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report_json", default=None)
    args = parser.parse_args()

    full_metadata = Path(args.full_metadata).expanduser().resolve()
    primary_clean_ids = Path(args.primary_clean_ids).expanduser().resolve()
    extra_clean_ids = Path(args.extra_clean_ids).expanduser().resolve() if args.extra_clean_ids else None
    output_path = Path(args.output).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve() if args.report_json else None

    full_records = load_jsonl(full_metadata)
    allowed_ids = load_id_set(primary_clean_ids)
    extra_ids_count = 0
    if extra_clean_ids and extra_clean_ids.exists():
        extra_ids = load_id_set(extra_clean_ids)
        allowed_ids |= extra_ids
        extra_ids_count = len(extra_ids)

    kept = [record for record in full_records if record["id"] in allowed_ids]
    save_jsonl(output_path, kept)

    report = {
        "full_metadata": str(full_metadata),
        "primary_clean_ids": str(primary_clean_ids),
        "extra_clean_ids": str(extra_clean_ids) if extra_clean_ids else None,
        "full_records": len(full_records),
        "primary_clean_count": len(load_id_set(primary_clean_ids)),
        "extra_clean_count": extra_ids_count,
        "output_records": len(kept),
        "output": str(output_path),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
