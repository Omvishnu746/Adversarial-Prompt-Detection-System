"""
PromptGuard – Structured JSON Logger

Writes one JSON object per line to logs/promptguard.log.
Each entry is human-readable and machine-parseable.

Log schema (Phase 2 update):
  {
    "timestamp":       ISO-8601 UTC string,
    "prompt":          first 200 chars of the original prompt,
    "risk_score":      float 0.0–1.0,
    "decision":        "ALLOW" | "BLOCK",
    "triggered_layer": "rule" | "semantic" | "none",
    "semantic_score":  float 0.0–1.0 | null,
    "semantic_match":  bool | null
  }
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import LOG_FILE_PATH


# ── Internal logging setup ────────────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    """Create and return a module-level logger backed by a JSON file handler."""
    log_path = Path(LOG_FILE_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("promptguard")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler – one JSON record per line
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        # Console handler – human-readable mirror
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s – %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(console_handler)

    return logger


_logger = _build_logger()


# ── Public API ────────────────────────────────────────────────────────────────

def log_request(
    prompt: str,
    risk_score: float,
    decision: str,
    triggered_layer: str,
    semantic_score: float | None = None,
    semantic_match: bool | None = None,
) -> None:
    """
    Append a structured JSON log entry for a single /check_prompt request.

    Args:
        prompt:          Raw user prompt (truncated to 200 chars in the log).
        risk_score:      Final risk score assigned to the prompt.
        decision:        "ALLOW" or "BLOCK".
        triggered_layer: "rule", "semantic", or "none".
        semantic_score:  Cosine similarity score from Phase 2 (None = engine off).
        semantic_match:  Whether the semantic layer triggered (None = engine off).
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt[:200],           # truncate to avoid massive log lines
        "risk_score": round(risk_score, 4),
        "decision": decision,
        "triggered_layer": triggered_layer,
        "semantic_score": round(semantic_score, 4) if semantic_score is not None else None,
        "semantic_match": semantic_match,
    }
    _logger.info(json.dumps(entry, ensure_ascii=False))
