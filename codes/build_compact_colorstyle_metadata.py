from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


SECTION_KEYS = [
    "GLOBAL_SUMMARY",
    "AGE_APPEARANCE",
    "PIGMENTATION",
    "MICROANEURYSMS",
    "HEMORRHAGES",
    "EXUDATES",
    "SOFT_EXUDATES",
    "MACULA_FOVEA",
    "OPTIC_DISC",
    "VESSELS",
]

KEEP_KEYS = [
    "MICROANEURYSMS",
    "HEMORRHAGES",
    "EXUDATES",
    "SOFT_EXUDATES",
    "MACULA_FOVEA",
    "OPTIC_DISC",
    "VESSELS",
]

DROP_ORDER = [
    "IMPRESSION",
    "VESSELS",
    "OPTIC_DISC",
    "MACULA_FOVEA",
]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def extract_sections(caption_base: str) -> tuple[dict[str, str], str]:
    text = normalize_whitespace(caption_base)
    if text.startswith("In the fundus photograph, "):
        text = text[len("In the fundus photograph, ") :]

    matches = list(re.finditer(r"(GLOBAL_SUMMARY|AGE_APPEARANCE|PIGMENTATION|MICROANEURYSMS|HEMORRHAGES|EXUDATES|SOFT_EXUDATES|MACULA_FOVEA|OPTIC_DISC|VESSELS):", text))
    if not matches:
        return {}, ""

    sections: dict[str, str] = {}
    impression = ""
    for idx, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = text[start:end].strip(" .;")
        if key == "VESSELS":
            split_marker = "These findings are consistent with"
            if split_marker in value:
                before, after = value.split(split_marker, 1)
                value = before.strip(" .;")
                impression = ("These findings are consistent with " + after.strip(" .;")).strip()
        sections[key] = value
    return sections, impression


def style_prefix(record: dict) -> str:
    code = str(record.get("color_style_code") or "").strip()
    prompt = str(record.get("color_style_prompt") or "").strip()
    if prompt.startswith("overall fundus tone is "):
        prompt = prompt[len("overall fundus tone is ") :]
    prompt = prompt.replace(" with a ", ", ").replace(" with ", ", ").replace(" and ", ", ")
    prompt = normalize_whitespace(prompt).strip(" .")
    if code and prompt:
        return f"COLOR_STYLE: {code}, {prompt}."
    if prompt:
        return f"COLOR_STYLE: {prompt}."
    if code:
        return f"COLOR_STYLE: {code}."
    return ""


def build_compact_base(record: dict) -> str:
    caption_base = str(record.get("caption_base") or record.get("caption") or "").strip()
    sections, impression = extract_sections(caption_base)
    parts: list[str] = []
    for key in KEEP_KEYS:
        value = sections.get(key)
        if value:
            parts.append(f"{key}: {value.strip(' .')}.")
    if impression:
        parts.append(f"IMPRESSION: {impression.strip(' .')}.")
    return normalize_whitespace(" ".join(parts))


def count_tokens(tokenizer, text: str) -> int:
    return int(tokenizer(text, return_tensors="pt").input_ids.shape[1])


def enforce_token_budget(tokenizer, style_text: str, compact_base: str, max_tokens: int) -> tuple[str, int]:
    sections = {}
    for key in KEEP_KEYS + ["IMPRESSION"]:
        marker = f"{key}:"
        if marker in compact_base:
            start = compact_base.index(marker)
            next_positions = [compact_base.find(f"{other}:", start + len(marker)) for other in KEEP_KEYS + ["IMPRESSION"] if other != key]
            next_positions = [pos for pos in next_positions if pos != -1]
            end = min(next_positions) if next_positions else len(compact_base)
            sections[key] = compact_base[start:end].strip()

    ordered = [key for key in KEEP_KEYS + ["IMPRESSION"] if key in sections]
    final_caption = normalize_whitespace(" ".join(x for x in [style_text, compact_base] if x))
    token_count = count_tokens(tokenizer, final_caption)
    if token_count <= max_tokens:
        return final_caption, token_count

    keep_set = set(ordered)
    for key in DROP_ORDER:
        keep_set.discard(key)
        rebuilt = " ".join(sections[k] for k in ordered if k in keep_set)
        final_caption = normalize_whitespace(" ".join(x for x in [style_text, rebuilt] if x))
        token_count = count_tokens(tokenizer, final_caption)
        if token_count <= max_tokens:
            return final_caption, token_count

    reduced_parts = []
    for key in ["MICROANEURYSMS", "HEMORRHAGES", "EXUDATES", "SOFT_EXUDATES"]:
        value = sections.get(key)
        if value:
            reduced_parts.append(value)
    final_caption = normalize_whitespace(" ".join(x for x in [style_text, " ".join(reduced_parts)] if x))
    token_count = count_tokens(tokenizer, final_caption)
    return final_caption, token_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build compact color-style metadata with color-first captions.")
    parser.add_argument("--metadata", type=str, nargs="+", required=True)
    parser.add_argument("--tokenizer_path", type=str, default="./google_gemma-2b")
    parser.add_argument("--output_suffix", type=str, default="_compact")
    parser.add_argument("--max_tokens", type=int, default=240)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(Path(args.tokenizer_path).expanduser().resolve())

    for metadata_arg in args.metadata:
        metadata_path = Path(metadata_arg).expanduser().resolve()
        rows = load_jsonl(metadata_path)
        compact_rows: list[dict] = []
        token_counts: list[int] = []
        for row in rows:
            style_text = style_prefix(row)
            compact_base = build_compact_base(row)
            compact_caption, token_count = enforce_token_budget(
                tokenizer,
                style_text=style_text,
                compact_base=compact_base,
                max_tokens=int(args.max_tokens),
            )
            token_counts.append(token_count)

            updated = dict(row)
            updated["caption_full_original"] = row.get("caption")
            updated["caption_base_full"] = row.get("caption_base") or row.get("caption")
            updated["caption_base"] = compact_base
            updated["caption"] = compact_caption
            updated["caption_compact_tokens"] = token_count
            updated["caption_compact_source"] = "build_compact_colorstyle_metadata.py"
            compact_rows.append(updated)

        output_path = metadata_path.with_name(metadata_path.stem + str(args.output_suffix) + metadata_path.suffix)
        write_jsonl(output_path, compact_rows)

        summary = {
            "source_metadata": str(metadata_path),
            "output_metadata": str(output_path),
            "count": len(compact_rows),
            "max_tokens": max(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "mean_tokens": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
        }
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] wrote {output_path} ({len(compact_rows)} records)")
        print(f"[ok] wrote {summary_path}")


if __name__ == "__main__":
    main()
