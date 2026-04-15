"""
PromptGuard – Global Settings
Centralised configuration loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env if present ────────────────────────────────────────────────────
load_dotenv()

# ── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parents[2]   # project root: .../promptguard/

# ── Dataset paths ────────────────────────────────────────────────────────────
RAW_DATASET_PATH: Path = Path(
    os.getenv(
        "RAW_DATASET_PATH",
        str(BASE_DIR.parent / "PromptGuard" / "Dataset" / "PromptGuard_Dataset_FINAL.json"),
    )
)

PROCESSED_DATASET_PATH: Path = Path(
    os.getenv(
        "PROCESSED_DATASET_PATH",
        str(BASE_DIR / "data" / "processed" / "cleaned_dataset.jsonl"),
    )
)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE_PATH: Path = Path(
    os.getenv("LOG_FILE_PATH", str(BASE_DIR / "logs" / "promptguard.log"))
)

# ── API settings ─────────────────────────────────────────────────────────────
API_TITLE: str = "PromptGuard"
API_VERSION: str = "0.1.0"
API_DESCRIPTION: str = (
    "Phase 1 – Rule-based adversarial prompt detection middleware."
)

# ── Risk thresholds ───────────────────────────────────────────────────────────
RULE_BLOCK_SCORE: float = 1.0
RULE_ALLOW_SCORE: float = 0.0
