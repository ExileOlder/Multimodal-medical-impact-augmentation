import json
from pathlib import Path


ROOT = Path("/root/autodl-tmp/retina/retina-text2cfp/codes")
TRAIN_ROOT = ROOT / "data" / "train" / "diabetic"
TEST_ROOT = ROOT / "data" / "test" / "diabetic"
MERGED_ROOT = ROOT / "data" / "merged" / "diabetic"


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def file_status(path: Path) -> dict:
    return {
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    train_autodl = TRAIN_ROOT / "autodl"
    test_autodl = TEST_ROOT / "autodl"
    merged_autodl = MERGED_ROOT / "autodl"

    result = {
        "train": {
            "wrappers_remaining": sorted(p.name for p in (TRAIN_ROOT / "source").glob("*.zip")),
            "images": count_images(TRAIN_ROOT / "source" / "train"),
            "pipeline_log": file_status(train_autodl / "pipeline.log"),
            "finalize_log": file_status(train_autodl / "finalize.log"),
            "metadata_all": file_status(train_autodl / "metadata_all.jsonl"),
            "metadata_train_full": file_status(train_autodl / "metadata_train_full.jsonl"),
            "metadata_example_20": file_status(train_autodl / "metadata_example_20.jsonl"),
            "excluded_records": file_status(train_autodl / "excluded_records.jsonl"),
            "build_report": load_report(train_autodl / "build_report.json"),
            "audit_report": load_report(train_autodl / "audit_train" / "audit_report.json"),
        },
        "test": {
            "wrappers_remaining": sorted(p.name for p in (TEST_ROOT / "source").glob("*.zip")),
            "images": count_images(TEST_ROOT / "source" / "test"),
            "pipeline_log": file_status(test_autodl / "pipeline.log"),
        },
        "merged": {
            "metadata_all": file_status(merged_autodl / "metadata_all.jsonl"),
            "metadata_train_full": file_status(merged_autodl / "metadata_train_full.jsonl"),
            "metadata_example_20": file_status(merged_autodl / "metadata_example_20.jsonl"),
            "excluded_records": file_status(merged_autodl / "excluded_records.jsonl"),
            "build_report": load_report(merged_autodl / "build_report.json"),
            "audit_report": load_report(merged_autodl / "audit_train" / "audit_report.json"),
        },
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
