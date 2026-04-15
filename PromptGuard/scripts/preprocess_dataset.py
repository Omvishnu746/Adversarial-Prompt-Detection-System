"""
PromptGuard - Dataset Preprocessing Script (Phase 1)

Usage:
    python scripts/preprocess_dataset.py
    python scripts/preprocess_dataset.py --input path/to/raw.json --output path/to/out.jsonl

What it does:
    1. Loads PromptGuard_Dataset_FINAL.json  (JSON array format)
    2. Cleans each prompt through the full preprocessing pipeline
    3. Saves the cleaned records to data/processed/cleaned_dataset.jsonl
    4. Prints a summary to stdout
"""

import sys
import argparse

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

# ── Make sure the project root is on sys.path when run as a script ────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from data.dataset_loader import load_dataset, save_cleaned_dataset
from app.services.preprocessing import clean_text, detect_base64, normalize_leetspeak
from app.config.settings import RAW_DATASET_PATH, PROCESSED_DATASET_PATH


# ── Preprocessing logic ────────────────────────────────────────────────────────

def preprocess_record(record: dict) -> dict:
    """
    Apply the full cleaning pipeline to a single dataset record.

    Steps applied to the ``prompt`` field:
      1. Unicode normalisation + lowercase + whitespace collapse
      2. Base64 detection (decoded text replaces the prompt if found)
      3. Leetspeak normalisation

    The ``attack_labels`` are preserved unchanged.
    """
    raw_prompt: str = record.get("prompt", "")

    # Step 1 – basic text cleaning
    cleaned = clean_text(raw_prompt)

    # Step 2 – base64 detection + inline replacement
    decoded = detect_base64(cleaned)
    if decoded:
        cleaned = clean_text(decoded)

    # Step 3 – leetspeak normalisation
    cleaned = normalize_leetspeak(cleaned)

    return {
        "prompt_original": raw_prompt,          # preserve raw for audit
        "prompt_cleaned": cleaned,
        "attack_labels": record.get("attack_labels", {}),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PromptGuard Phase 1 – Dataset Preprocessor")
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DATASET_PATH),
        help="Path to the raw JSON dataset file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DATASET_PATH),
        help="Path to write the cleaned JSONL file.",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  PromptGuard – Phase 1 Dataset Preprocessor")
    print(f"{'='*60}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"{'='*60}\n")

    # ── 1. Load raw dataset ───────────────────────────────────────────────────
    print("[1/3] Loading raw dataset...")
    try:
        raw_records = load_dataset(args.input)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    total = len(raw_records)
    print(f"      Loaded {total:,} records.\n")

    # ── 2. Clean records ──────────────────────────────────────────────────────
    print("[2/3] Cleaning records...")
    cleaned_records = []
    skipped = 0

    for record in tqdm(raw_records, desc="Preprocessing", unit="record"):
        if not record.get("prompt"):
            skipped += 1
            continue
        cleaned_records.append(preprocess_record(record))

    print(f"\n      Cleaned : {len(cleaned_records):,} records")
    print(f"      Skipped : {skipped:,} records (empty prompt)\n")

    # ── 3. Save processed dataset ─────────────────────────────────────────────
    print("[3/3] Saving cleaned dataset...")
    out_path = save_cleaned_dataset(cleaned_records, args.output)
    print(f"      Saved -> {out_path}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  [OK] Preprocessing complete.")
    print(f"      Total processed : {len(cleaned_records):,} / {total:,} records")
    print(f"      Output file     : {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
