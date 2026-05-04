"""
PromptGuard – Router Configuration (Phase 4)

Central location for all decision-routing thresholds.
Tune these values without touching any business logic.

Threshold semantics
───────────────────
RULE_BLOCK_THRESHOLD   Rule engine score at or above which the router hard-blocks,
                        regardless of other signals.  Rules are deterministic, so this
                        can safely be lower than the aggregation threshold.

CHUNK_BLOCK_THRESHOLD  Maximum chunk adversarial probability at or above which the
                        router hard-blocks.  A single highly adversarial chunk in a
                        long prompt should be treated as a critical signal.

BLOCK_THRESHOLD        Aggregated final_risk_score at or above which the router
                        decides BLOCK (after rule and chunk checks).

ALLOW_THRESHOLD        Aggregated final_risk_score at or below which the router
                        decides ALLOW.  Scores in the range
                        (ALLOW_THRESHOLD, BLOCK_THRESHOLD) trigger SANITIZE.
"""

# ── Rule engine ───────────────────────────────────────────────────────────────
RULE_BLOCK_THRESHOLD: float = 0.90

# ── Chunk-level risk ─────────────────────────────────────────────────────────
# Must match the classifier engine's ADVERSARIAL_THRESHOLD (0.99999).
# Keeping these in sync ensures the chunk check and classifier always agree
# on what constitutes an adversarial signal.
CHUNK_BLOCK_THRESHOLD: float = 0.99999

# ── Aggregated risk ───────────────────────────────────────────────────────────
BLOCK_THRESHOLD: float = 0.70
ALLOW_THRESHOLD: float = 0.40
