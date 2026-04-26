"""Clean raw VLM-generated reports for use as SGProtoNet semantic inputs.

Removes chatty LLM preambles, markdown formatting, and section headers so
that BiomedCLIP's PubMedBERT text encoder receives clean medical prose.

Transformations applied:
  1. Strip leading preamble lines  ("Here's a description...", "Based on the image,")
  2. Strip **Pathological Description:** section header (keep its content)
  3. Remove all **bold** markers
  4. Collapse newlines → single space
  5. Drop truncated last sentence if it ends without punctuation

Usage:
    python scripts/clean_reports.py \\
        --input_csv  data/reports/medgemma_reports.csv \\
        --output_csv data/reports/medgemma_reports_clean.csv

    # Preview first 5 rows without writing:
    python scripts/clean_reports.py \\
        --input_csv data/reports/medgemma_reports.csv \\
        --preview
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Cleaning logic
# ---------------------------------------------------------------------------

# Leading preamble patterns (case-insensitive, anchored to start of text)
_PREAMBLES = [
    r"^Here['\u2019]s a description of the image based on the provided information:\s*",
    r"^Here['\u2019]s a pathological description.*?:\s*",
    r"^Based on the (?:provided |H&E stained )?image[,.]?\s*",
    r"^Based on the (?:provided )?information[,.]?\s*",
    r"^The following is a (?:pathological )?description.*?:\s*",
]

_PREAMBLE_RE = re.compile(
    "|".join(_PREAMBLES),
    flags=re.IGNORECASE | re.DOTALL,
)

# Section header: **Pathological Description:** — remove header, keep body
_SECTION_HEADER_RE = re.compile(
    r"\*\*\s*Pathological Description\s*:\s*\*\*\s*",
    flags=re.IGNORECASE,
)

# Bold markers: **text** → text
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)

# Bullet / list markers
_BULLET_RE = re.compile(r"^\s*[\*\-•]\s+", flags=re.MULTILINE)

# Multiple whitespace / newlines → single space
_WHITESPACE_RE = re.compile(r"\s+")


def clean_report(text: str) -> str:
    """Apply all cleaning steps and return a single-line medical description."""
    if not text or text.startswith("ERROR") or text.startswith("LOAD_ERROR"):
        return text

    # 1. Strip leading preamble
    text = _PREAMBLE_RE.sub("", text.strip())

    # 2. Remove **Pathological Description:** header (keep content after it)
    text = _SECTION_HEADER_RE.sub(" ", text)

    # 3. Remove **bold** markers, keep inner text
    text = _BOLD_RE.sub(r"\1", text)

    # 4. Remove bullet/list markers
    text = _BULLET_RE.sub("", text)

    # 5. Collapse all whitespace (newlines, tabs, multiple spaces) → single space
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # 6. Drop truncated final sentence (no ending punctuation)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences and not re.search(r"[.!?]$", sentences[-1]):
        sentences = sentences[:-1]

    # 7. Deduplicate repeated sentences (MedGemma sometimes loops)
    seen: set[str] = set()
    deduped: list[str] = []
    for s in sentences:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)
    text = " ".join(deduped).strip()

    # 8. Capitalise first character
    if text:
        text = text[0].upper() + text[1:]

    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean VLM reports CSV for SGProtoNet training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_csv",  required=True, help="Raw reports CSV.")
    p.add_argument("--output_csv", default=None,
                   help="Output path. Defaults to <input>_clean.csv")
    p.add_argument("--preview", action="store_true",
                   help="Print first 5 cleaned reports and exit (no file written).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)

    with input_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Input CSV is empty.")
        return

    fieldnames = list(rows[0].keys())

    if args.preview:
        print(f"Total rows: {len(rows)}\n")
        for row in rows[:5]:
            raw = row.get("report", "")
            cleaned = clean_report(raw)
            print(f"[{row.get('subtype','?')} | {row.get('magnification','?')}]")
            print(f"  RAW    : {raw[:120].replace(chr(10),' ')!r}")
            print(f"  CLEANED: {cleaned[:120]!r}")
            print()
        return

    output_csv = Path(args.output_csv) if args.output_csv else \
        input_csv.with_stem(input_csv.stem + "_clean")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cleaned_rows = []
    empty_after_clean = 0
    for row in rows:
        row["report"] = clean_report(row.get("report", ""))
        if not row["report"]:
            empty_after_clean += 1
        cleaned_rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"Cleaned {len(cleaned_rows)} rows → {output_csv}")
    if empty_after_clean:
        print(f"Warning: {empty_after_clean} rows have empty reports after cleaning.")


if __name__ == "__main__":
    main()
