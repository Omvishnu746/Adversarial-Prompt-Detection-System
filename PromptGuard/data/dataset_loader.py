"""
PromptGuard – Dataset Loader

Handles loading the raw PromptGuard_Dataset_FINAL.json (a JSON array)
and saving the cleaned output as JSONL (one JSON object per line).

The dataset schema per record:
  {
    "prompt": str,
    "attack_labels": {
      "is_benign":              int,   # 1 = safe, 0 = adversarial
      "intent_jailbreak":       int,
      "intent_prompt_injection": int,
      "tech_roleplay":          int,
      "tech_indirect":          int,
      "tech_obfuscation":       int
    }
  }
"""

import json
from pathlib import Path
from typing import Any

from app.config.settings import RAW_DATASET_PATH, PROCESSED_DATASET_PATH


# ── Load ──────────────────────────────────────────────────────────────────────

def load_dataset(path: Path | str | None = None) -> list[dict[str, Any]]:
    """
    Load the raw JSON array dataset from disk.

    Args:
        path: Override the default RAW_DATASET_PATH from settings.

    Returns:
        List of raw record dicts.

    Raises:
        FileNotFoundError: if the dataset file does not exist.
        ValueError:        if the file is not a valid JSON array.
    """
    dataset_path = Path(path) if path else Path(RAW_DATASET_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            "Set RAW_DATASET_PATH in your .env or pass the path explicitly."
        )

    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at the top level, got {type(data).__name__}."
        )

    return data


# ── Save ──────────────────────────────────────────────────────────────────────

def save_cleaned_dataset(
    records: list[dict[str, Any]],
    path: Path | str | None = None,
) -> Path:
    """
    Write cleaned records to a JSONL file (one JSON object per line).

    Args:
        records: List of cleaned record dicts.
        path:    Override the default PROCESSED_DATASET_PATH from settings.

    Returns:
        Resolved Path of the written file.
    """
    out_path = Path(path) if path else Path(PROCESSED_DATASET_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path


# ── Load processed ────────────────────────────────────────────────────────────

def load_cleaned_dataset(path: Path | str | None = None) -> list[dict[str, Any]]:
    """
    Load the processed JSONL dataset from disk.

    Args:
        path: Override the default PROCESSED_DATASET_PATH from settings.

    Returns:
        List of cleaned record dicts.
    """
    in_path = Path(path) if path else Path(PROCESSED_DATASET_PATH)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {in_path}\n"
            "Run: python scripts/preprocess_dataset.py"
        )

    records: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records
