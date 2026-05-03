"""
PromptGuard – Decision Router Engine (Phase 4 / Tier 4: The Bouncer)

This is the final enforcement layer before a prompt reaches an LLM.
It receives the fully aggregated risk signals from Phase 3.5 and produces
one of three routing decisions:

  ALLOW     – Prompt is safe. Pass it through unchanged.
  BLOCK     – Prompt is adversarial. Reject it entirely.
  SANITIZE  – Prompt is borderline. Strip adversarial patterns and pass
               the cleaned text downstream.

Decision Priority
─────────────────
1. Rule-based hard block  (rule_score >= RULE_BLOCK_THRESHOLD)
2. Chunk-based hard block (max_chunk_score >= CHUNK_BLOCK_THRESHOLD)
3. Aggregated risk block  (final_risk_score >= BLOCK_THRESHOLD)
4. Safe allow             (final_risk_score <= ALLOW_THRESHOLD)
5. Sanitize fallback      (moderate risk in the grey zone)

Performance
───────────
The router performs no heavy computation (no ML inference, no I/O).
All operations are pure regex + arithmetic, well within the < 2 ms target.
"""

import logging
import math
import re
from typing import Optional

from app.models.router_response import RouterResponse
from app.config.router_config import (
    BLOCK_THRESHOLD,
    ALLOW_THRESHOLD,
    CHUNK_BLOCK_THRESHOLD,
    RULE_BLOCK_THRESHOLD,
)

logger = logging.getLogger("promptguard.router_engine")

# ── Sanitization patterns ─────────────────────────────────────────────────────
# Each tuple is (compiled_regex, replacement_string).
# Order matters: more specific patterns first.
_SANITIZE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Instruction override / injection
    (re.compile(r"ignore\s+(all\s+)?(?:previous|prior|above|earlier)\s+instructions?", re.I), "[REDACTED]"),
    (re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?", re.I), "[REDACTED]"),
    (re.compile(r"forget\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?", re.I), "[REDACTED]"),
    (re.compile(r"override\s+(?:system|safety|all)?\s*(?:instructions?|rules?|prompt)", re.I), "[REDACTED]"),
    (re.compile(r"bypass\s+(?:safety|security|content|all)?\s*(?:filter|guard|safeguard|restriction|rule)s?", re.I), "[REDACTED]"),

    # System prompt extraction
    (re.compile(r"reveal\s+(?:your\s+)?system\s+prompt", re.I), "[REDACTED]"),
    (re.compile(r"show\s+(?:me\s+)?(?:your\s+)?(?:system|hidden|internal)\s+(?:prompt|instructions?)", re.I), "[REDACTED]"),
    (re.compile(r"print\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)", re.I), "[REDACTED]"),
    (re.compile(r"output\s+(?:your\s+)?(?:system|original)\s+(?:prompt|instructions?)", re.I), "[REDACTED]"),

    # Jailbreak persona
    (re.compile(r"\bact\s+as\s+(?:a\s+)?(?:DAN|an?\s+AI\s+with(?:out)?\s+(?:restrictions?|limits?|guidelines?))", re.I), "[REDACTED]"),
    (re.compile(r"\bpretend\s+(?:you\s+(?:are|have)\s+)?(?:no\s+(?:rules?|restrictions?|guidelines?|ethics?))", re.I), "[REDACTED]"),
    (re.compile(r"\byou\s+are\s+now\s+(?:DAN|an?\s+AI\s+without)", re.I), "[REDACTED]"),
    (re.compile(r"\bdo\s+anything\s+now\b", re.I), "[REDACTED]"),
    (re.compile(r"\bno\s+(?:ethical\s+)?(?:restrictions?|guidelines?|limits?|filters?)\b", re.I), "[REDACTED]"),

    # System override / jailbreak keywords
    (re.compile(r"\bsystem\s+override\b", re.I), "[REDACTED]"),
    (re.compile(r"\bjailbreak\b", re.I), "[REDACTED]"),
    (re.compile(r"\bunrestricted\s+mode\b", re.I), "[REDACTED]"),
    (re.compile(r"\bdeveloper\s+mode\b", re.I), "[REDACTED]"),
]

# Minimum length of sanitized text (guards against prompts that are entirely
# adversarial pattern and would become empty after cleaning).
_MIN_SANITIZED_LENGTH: int = 3


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp and validate a score to [lo, hi], handling NaN/None gracefully."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return max(lo, min(hi, float(value)))


def sanitize_text(text: str) -> str:
    """
    Remove known adversarial prompt patterns from *text* using regex.

    Rules
    ─────
    • Each matched pattern is replaced with '[REDACTED]'.
    • Multiple consecutive '[REDACTED]' tokens are collapsed to one.
    • If sanitization would reduce the text below _MIN_SANITIZED_LENGTH
      characters (i.e., the entire prompt was adversarial), the original
      text is returned with a warning prefix instead of an empty string.
      This prevents a sanitized prompt from being an empty request.

    Args:
        text: The raw or preprocessed prompt string.

    Returns:
        The sanitized prompt string.
    """
    cleaned = text
    for pattern, replacement in _SANITIZE_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)

    # Collapse repeated [REDACTED] blocks
    cleaned = re.sub(r"(\[REDACTED\]\s*){2,}", "[REDACTED] ", cleaned).strip()

    if len(cleaned.replace("[REDACTED]", "").strip()) < _MIN_SANITIZED_LENGTH:
        logger.warning(
            "Sanitization removed almost all content from prompt (len=%d). "
            "Returning prefixed original to avoid empty downstream request.",
            len(text),
        )
        return "[SANITIZED – CONTENT REMOVED] " + text[:120]

    return cleaned


def route_decision(
    final_risk_score: float,
    max_chunk_score:  float,
    rule_score:       float,
    original_prompt:  str,
) -> RouterResponse:
    """
    Apply the Tier 4 routing logic and return a structured RouterResponse.

    Decision priority (highest to lowest):
      1. Rule-based hard block
      2. Chunk-level hard block
      3. Aggregated risk block
      4. Safe allow
      5. Sanitize (grey zone)

    Args:
        final_risk_score: Weighted aggregated score from Phase 3.5 (0.0–1.0).
        max_chunk_score:  Maximum adversarial probability across all chunks (0.0–1.0).
        rule_score:       Rule engine output — 1.0 if any rule matched, else 0.0.
        original_prompt:  The raw prompt string (used for sanitization).

    Returns:
        RouterResponse with decision, confidence, reason, and optional sanitized_text.
    """
    # ── Input sanitisation ────────────────────────────────────────────────────
    r  = _clamp(rule_score)
    k  = _clamp(max_chunk_score)
    f  = _clamp(final_risk_score)

    sanitized_text: Optional[str] = None

    # ── Priority 1: Rule-based hard block ─────────────────────────────────────
    if r >= RULE_BLOCK_THRESHOLD:
        logger.debug("Router: BLOCK via rule (rule_score=%.4f >= %.2f)", r, RULE_BLOCK_THRESHOLD)
        return RouterResponse(
            decision="BLOCK",
            confidence=round(f, 4),
            reason="Rule-based critical pattern detected",
            sanitized_text=None,
        )

    # ── Priority 2: Chunk-based hard block ────────────────────────────────────
    if k >= CHUNK_BLOCK_THRESHOLD:
        logger.debug("Router: BLOCK via chunk (chunk_score=%.4f >= %.2f)", k, CHUNK_BLOCK_THRESHOLD)
        return RouterResponse(
            decision="BLOCK",
            confidence=round(f, 4),
            reason="High-risk chunk detected",
            sanitized_text=None,
        )

    # ── Priority 3: Aggregated risk block ────────────────────────────────────
    if f >= BLOCK_THRESHOLD:
        logger.debug("Router: BLOCK via aggregation (final_risk=%.4f >= %.2f)", f, BLOCK_THRESHOLD)
        return RouterResponse(
            decision="BLOCK",
            confidence=round(f, 4),
            reason="Aggregated risk exceeds block threshold",
            sanitized_text=None,
        )

    # ── Priority 4: Safe allow ────────────────────────────────────────────────
    if f <= ALLOW_THRESHOLD:
        logger.debug("Router: ALLOW (final_risk=%.4f <= %.2f)", f, ALLOW_THRESHOLD)
        return RouterResponse(
            decision="ALLOW",
            confidence=round(f, 4),
            reason="Low risk detected",
            sanitized_text=None,
        )

    # ── Priority 5: Sanitize (grey zone) ─────────────────────────────────────
    sanitized_text = sanitize_text(original_prompt)
    logger.debug(
        "Router: SANITIZE (final_risk=%.4f in grey zone [%.2f, %.2f])",
        f, ALLOW_THRESHOLD, BLOCK_THRESHOLD,
    )
    return RouterResponse(
        decision="SANITIZE",
        confidence=round(f, 4),
        reason="Moderate risk detected — adversarial patterns removed",
        sanitized_text=sanitized_text,
    )
