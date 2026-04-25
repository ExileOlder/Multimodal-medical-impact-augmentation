import argparse
import csv
import json
import os
import subprocess
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def get_kaggle_auth_mode() -> str | None:
    if bool(os.environ.get("KAGGLE_API_TOKEN", "").strip()):
        return "api_token"
    has_env = bool(os.environ.get("KAGGLE_USERNAME", "").strip()) and bool(os.environ.get("KAGGLE_KEY", "").strip())
    if has_env:
        return "username_key"
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return "kaggle_json"
    return None


def ensure_kaggle_credentials() -> str:
    auth_mode = get_kaggle_auth_mode()
    if auth_mode is not None:
        return auth_mode
    raise FileNotFoundError(
        "Kaggle credentials were not found. "
        "Provide KAGGLE_API_TOKEN, provide KAGGLE_USERNAME/KAGGLE_KEY, "
        "or place kaggle.json under ~/.kaggle/."
    )


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def run_command_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def normalize_requested_files(raw_files: list[str]) -> list[str]:
    requested: list[str] = []
    for value in raw_files:
        for item in value.split(","):
            item = item.strip()
            if item:
                requested.append(item)
    return requested or ["train.zip"]


def list_competition_files(competition: str) -> list[str]:
    output = run_command_capture(["kaggle", "competitions", "files", "-v", competition])
    reader = csv.DictReader(StringIO(output))
    return [row["name"] for row in reader if row.get("name")]


def resolve_requested_files(requested_files: list[str], available_files: list[str]) -> dict[str, list[str]]:
    available_set = set(available_files)
    resolved: dict[str, list[str]] = {}
    for requested in requested_files:
        if requested in available_set:
            resolved[requested] = [requested]
            continue
        multipart = sorted(
            name
            for name in available_files
            if name.startswith(f"{requested}.") and name[len(requested) + 1 :].isdigit()
        )
        if multipart:
            resolved[requested] = multipart
            continue
        raise FileNotFoundError(f"Requested Kaggle file was not found: {requested}")
    return resolved


def get_download_path(filename: str, archive_dir: Path) -> Path:
    if filename.lower().endswith(".zip"):
        return archive_dir / filename
    return archive_dir / f"{filename}.zip"


def materialize_downloaded_file(
    filename: str,
    download_path: Path,
    archive_dir: Path,
    force: bool,
    keep_download_wrapper: bool,
) -> tuple[Path, bool]:
    target_path = archive_dir / filename
    if download_path == target_path:
        return target_path, False
    if force and target_path.exists():
        target_path.unlink()
    if not target_path.exists():
        with zipfile.ZipFile(download_path) as handle:
            handle.extractall(archive_dir)
    if not target_path.exists():
        raise FileNotFoundError(f"Downloaded wrapper archive did not yield expected file: {target_path}")
    if not keep_download_wrapper:
        download_path.unlink(missing_ok=True)
    return target_path, True


def extract_zip(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as handle:
        handle.extractall(output_dir)


def combine_multipart_archives(part_paths: list[Path], merged_archive_path: Path, remove_parts: bool) -> None:
    with merged_archive_path.open("wb") as merged:
        for part_path in part_paths:
            with part_path.open("rb") as source:
                while True:
                    chunk = source.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    merged.write(chunk)
            if remove_parts:
                part_path.unlink(missing_ok=True)


def download_one_file(
    filename: str,
    competition: str,
    archive_dir: Path,
    force: bool,
    keep_archives: bool,
) -> dict:
    download_path = get_download_path(filename, archive_dir)
    if force and download_path.exists():
        download_path.unlink()

    downloaded = False
    if not download_path.exists():
        run_command(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition,
                "-f",
                filename,
                "-p",
                str(archive_dir),
                "-q",
            ]
        )
        downloaded = True

    usable_path, materialized = materialize_downloaded_file(
        filename=filename,
        download_path=download_path,
        archive_dir=archive_dir,
        force=force,
        keep_download_wrapper=keep_archives,
    )
    return {
        "filename": filename,
        "download_path": download_path,
        "usable_path": usable_path,
        "downloaded": downloaded,
        "materialized": materialized,
    }


def summarize_images(root: Path, sample_size: int) -> dict:
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    image_paths = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
    suffix_counter = Counter(p.suffix.lower() for p in image_paths)
    total_size = sum(p.stat().st_size for p in image_paths)
    return {
        "root_dir": str(root),
        "image_count": len(image_paths),
        "total_image_bytes": total_size,
        "suffix_counts": dict(suffix_counter),
        "sample_files": [str(p.relative_to(root)) for p in image_paths[:sample_size]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download diabetic retinopathy fundus images from Kaggle.")
    parser.add_argument(
        "--competition",
        type=str,
        default="diabetic-retinopathy-detection",
        help="Kaggle competition slug.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store extracted real fundus images.",
    )
    parser.add_argument(
        "--archive_dir",
        type=str,
        default="",
        help="Directory to store downloaded zip archives. Defaults to output_dir/_archives.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Competition archive to download, e.g. train.zip. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--keep_archives",
        action="store_true",
        help="Keep downloaded archives after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives even if they already exist locally.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="Number of sample extracted files to include in the report.",
    )
    parser.add_argument(
        "--parallel_downloads",
        type=int,
        default=2,
        help="Number of Kaggle file downloads to run concurrently.",
    )
    args = parser.parse_args()

    auth_mode = ensure_kaggle_credentials()

    output_dir = Path(args.output_dir).expanduser().resolve()
    archive_dir = Path(args.archive_dir).expanduser().resolve() if args.archive_dir else output_dir / "_archives"
    requested_files = normalize_requested_files(args.file)
    available_files = list_competition_files(args.competition)
    resolved_files = resolve_requested_files(requested_files, available_files)

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    downloaded_archives: list[str] = []
    materialized_archives: list[str] = []
    extracted_archives: list[str] = []
    combined_archives: list[str] = []
    download_path_map: dict[str, Path] = {}
    usable_path_map: dict[str, Path] = {}
    unique_downloads: list[str] = []
    for filenames in resolved_files.values():
        for filename in filenames:
            if filename not in unique_downloads:
                unique_downloads.append(filename)

    max_workers = max(1, args.parallel_downloads)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                download_one_file,
                filename=filename,
                competition=args.competition,
                archive_dir=archive_dir,
                force=args.force,
                keep_archives=args.keep_archives,
            ): filename
            for filename in unique_downloads
        }
        for future in as_completed(future_map):
            result = future.result()
            filename = result["filename"]
            download_path_map[filename] = result["download_path"]
            usable_path_map[filename] = result["usable_path"]
            if result["downloaded"]:
                downloaded_archives.append(str(result["download_path"]))
            if result["materialized"]:
                materialized_archives.append(str(result["usable_path"]))

    for requested, filenames in resolved_files.items():
        if len(filenames) == 1 and filenames[0].lower().endswith(".zip"):
            archive_path = usable_path_map[filenames[0]]
            extract_zip(archive_path, output_dir)
            extracted_archives.append(str(archive_path))
            if not args.keep_archives:
                archive_path.unlink(missing_ok=True)
            continue

        part_paths = [usable_path_map[filename] for filename in filenames]
        merged_archive_path = archive_dir / requested
        if args.force and merged_archive_path.exists():
            merged_archive_path.unlink()
        if not merged_archive_path.exists():
            combine_multipart_archives(part_paths, merged_archive_path, remove_parts=not args.keep_archives)
            combined_archives.append(str(merged_archive_path))
        extract_zip(merged_archive_path, output_dir)
        extracted_archives.append(str(merged_archive_path))
        if not args.keep_archives:
            merged_archive_path.unlink(missing_ok=True)

    report = {
        "competition": args.competition,
        "auth_mode": auth_mode,
        "requested_files": requested_files,
        "resolved_files": resolved_files,
        "downloaded_archives": downloaded_archives,
        "materialized_archives": materialized_archives,
        "combined_archives": combined_archives,
        "extracted_archives": extracted_archives,
        "keep_archives": args.keep_archives,
        "force": args.force,
        "parallel_downloads": max_workers,
        "summary": summarize_images(output_dir, args.sample_size),
    }
    report_path = output_dir / "download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print(f"Auth mode: {auth_mode}")
    print(f"Requested files: {requested_files}")
    print(f"Resolved files: {resolved_files}")
    print(f"Downloaded archives: {len(downloaded_archives)}")
    print(f"Parallel downloads: {max_workers}")
    print(f"Combined archives: {len(combined_archives)}")
    print(f"Extracted archives: {len(extracted_archives)}")
    print(f"Image count: {report['summary']['image_count']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
