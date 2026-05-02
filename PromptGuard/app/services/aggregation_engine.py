"""
PromptGuard – Risk Aggregation Engine (Phase 3.5)

Combines signals from all detection layers into a single, normalised
``final_risk_score`` using a fixed weighted average formula.

Weights (must sum to 1.0):
  Rule Engine  (deterministic)   → 0.30
  Classifier   (DistilBERT)      → 0.30
  Semantic     (SBERT + FAISS)   → 0.25
  Chunk Risk   (max chunk prob)  → 0.15

Design decisions
────────────────
• All input scores are clamped to [0.0, 1.0] before aggregation to guard
  against upstream float precision artefacts.
• The function is pure (no side-effects) and fully synchronous so it can be
  called from both sync and async contexts.
• Weights are defined as module-level constants so they can be tuned
  independently without touching the formula.
"""

import logging
from typing import TypedDict

logger = logging.getLogger("promptguard.aggregation_engine")

# ── Aggregation weights (must sum to 1.0) ────────────────────────────────────
WEIGHT_RULE:       float = 0.30
WEIGHT_CLASSIFIER: float = 0.30
WEIGHT_SEMANTIC:   float = 0.25
WEIGHT_CHUNK:      float = 0.15

assert abs(WEIGHT_RULE + WEIGHT_CLASSIFIER + WEIGHT_SEMANTIC + WEIGHT_CHUNK - 1.0) < 1e-9, (
    "Aggregation weights must sum to exactly 1.0"
)


class AggregationResult(TypedDict):
    """Typed output of aggregate_risk()."""
    rule_score:        float
    semantic_score:    float
    classifier_score:  float
    chunk_risk_score:  float
    final_risk_score:  float


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def aggregate_risk(
    rule_score:       float,
    semantic_score:   float,
    classifier_score: float,
    chunk_risk_score: float,
) -> AggregationResult:
    """
    Combine detection-layer signals into a single ``final_risk_score``.

    All inputs are clamped to [0.0, 1.0] before the weighted average is
    applied so upstream values outside that range do not corrupt the output.

    Formula
    -------
    final_risk_score = (
        0.30 * rule_score
      + 0.30 * classifier_score
      + 0.25 * semantic_score
      + 0.15 * chunk_risk_score
    )

    Args:
        rule_score:       1.0 if any rule matched, 0.0 otherwise.
        semantic_score:   Cosine similarity score from SBERT (0.0–1.0).
        classifier_score: Adversarial probability from DistilBERT (0.0–1.0).
        chunk_risk_score: Maximum adversarial probability across all chunks (0.0–1.0).

    Returns:
        AggregationResult dict with all five fields.

    Example
    -------
    >>> result = aggregate_risk(
    ...     rule_score=1.0, semantic_score=0.70,
    ...     classifier_score=0.80, chunk_risk_score=0.85
    ... )
    >>> round(result["final_risk_score"], 2)
    0.83
    """
    # 1. Clamp all inputs
    r = _clamp(rule_score)
    s = _clamp(semantic_score)
    c = _clamp(classifier_score)
    k = _clamp(chunk_risk_score)

    # 2. Weighted average
    final = (
        WEIGHT_RULE       * r
        + WEIGHT_CLASSIFIER * c
        + WEIGHT_SEMANTIC   * s
        + WEIGHT_CHUNK      * k
    )

    # 3. Clamp output (floating-point safety)
    final = _clamp(final)

    logger.debug(
        "aggregate_risk: rule=%.4f semantic=%.4f classifier=%.4f chunk=%.4f → final=%.4f",
        r, s, c, k, final,
    )

    return AggregationResult(
        rule_score=round(r, 4),
        semantic_score=round(s, 4),
        classifier_score=round(c, 4),
        chunk_risk_score=round(k, 4),
        final_risk_score=round(final, 4),
    )
