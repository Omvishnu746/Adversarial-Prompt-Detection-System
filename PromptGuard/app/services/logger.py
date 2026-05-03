"""
PromptGuard – Structured JSON Logger

Writes one JSON object per line to logs/promptguard.log.
Each entry is human-readable and machine-parseable.

Log schema (Phase 3.5 update):
  {
    "timestamp":            ISO-8601 UTC string,
    "prompt":               first 200 chars of the original prompt,
    "risk_score":           float 0.0–1.0,
    "decision":             "ALLOW" | "BLOCK",
    "triggered_layer":      "rule" | "semantic" | "classifier" | "aggregation" | "none",
    "semantic_score":       float 0.0–1.0 | null,
    "semantic_match":       bool | null,
    "classifier_score":     float 0.0–1.0 | null,
    "classifier_triggered": bool | null,
    "rule_score":           float 0.0–1.0 | null,
    "chunk_risk_score":     float 0.0–1.0 | null,
    "final_risk_score":     float 0.0–1.0 | null
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
    classifier_score: float | None = None,
    classifier_triggered: bool | None = None,
    rule_score: float | None = None,
    chunk_risk_score: float | None = None,
    final_risk_score: float | None = None,
    router_decision: str | None = None,
    router_reason: str | None = None,
    sanitized: bool | None = None,
) -> None:
    """
    Append a structured JSON log entry for a single /check_prompt request.

    Args:
        prompt:               Raw user prompt (truncated to 200 chars in the log).
        risk_score:           Final risk score assigned to the prompt.
        decision:             "ALLOW", "BLOCK", or "SANITIZE".
        triggered_layer:      "rule", "semantic", "classifier", "aggregation", or "none".
        semantic_score:       Cosine similarity score from Phase 2 (None = engine off).
        semantic_match:       Whether the semantic layer triggered (None = engine off).
        classifier_score:     Adversarial probability from Phase 3 (None = not loaded).
        classifier_triggered: Whether the classifier layer triggered (None = not loaded).
        rule_score:           Rule engine output score (None = not evaluated).
        chunk_risk_score:     Max adversarial probability across chunks (None = not available).
        final_risk_score:     Aggregated risk score from Phase 3.5 (None = not computed).
        router_decision:      Final Phase 4 router decision (None = router not run).
        router_reason:        Human-readable reason from the Phase 4 router.
        sanitized:            True if the prompt was sanitized before passing downstream.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt[:200],
        "risk_score": round(risk_score, 4),
        "decision": decision,
        "triggered_layer": triggered_layer,
        "semantic_score": round(semantic_score, 4) if semantic_score is not None else None,
        "semantic_match": semantic_match,
        "classifier_score": round(classifier_score, 4) if classifier_score is not None else None,
        "classifier_triggered": classifier_triggered,
        "rule_score": round(rule_score, 4) if rule_score is not None else None,
        "chunk_risk_score": round(chunk_risk_score, 4) if chunk_risk_score is not None else None,
        "final_risk_score": round(final_risk_score, 4) if final_risk_score is not None else None,
        "router_decision": router_decision,
        "router_reason": router_reason,
        "sanitized": sanitized,
    }
    _logger.info(json.dumps(entry, ensure_ascii=False))
