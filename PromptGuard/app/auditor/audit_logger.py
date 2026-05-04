"""
PromptGuard – Audit Logger (Phase 5)

Public API
──────────
trigger_audit_log(**kwargs)
    Fire-and-forget: dispatches log_attack_event as a Celery task.
    Returns immediately (< 1 ms).  The worker handles everything async.

    Falls back gracefully if Redis / Celery is unavailable so the API
    continues to work even without the auditor infrastructure running.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks

logger = logging.getLogger("promptguard.audit_logger")


def trigger_audit_log(
    *,
    background_tasks: BackgroundTasks,
    prompt: str,
    decision: str,
    triggered_layer: str = "none",
    rule_score: float = 0.0,
    semantic_score: float = 0.0,
    classifier_score: float = 0.0,
    chunk_risk_score: float = 0.0,
    final_risk_score: float = 0.0,
    sanitized: bool = False,
    target_llm: Optional[str] = None,
) -> None:
    """
    Dispatch an async audit task using FastAPI BackgroundTasks.

    This function is called from check_prompt.py immediately after the router
    decision. It enqueues the task and returns in < 1 ms.

    Parameters
    ----------
    prompt          Raw prompt text.
    decision        ALLOW | BLOCK | SANITIZE.
    triggered_layer Which detection layer fired (rule / semantic / classifier / aggregation / none).
    rule_score      Rule-engine score.
    semantic_score  SBERT similarity score.
    classifier_score  Raw DistilBERT adversarial probability.
    chunk_risk_score  Per-chunk adversarial probability.
    final_risk_score  Weighted aggregated score.
    sanitized       True when SANITIZE decision was taken.
    target_llm      Downstream LLM identifier (optional).
    """
    try:
        from app.auditor.audit_tasks import log_attack_event

        background_tasks.add_task(
            log_attack_event,
            prompt=prompt,
            decision=decision,
            triggered_layer=triggered_layer,
            rule_score=round(rule_score, 6),
            semantic_score=round(semantic_score, 6),
            classifier_score=round(classifier_score, 6),
            chunk_risk_score=round(chunk_risk_score, 6),
            final_risk_score=round(final_risk_score, 6),
            sanitized=sanitized,
            target_llm=target_llm,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as exc:
        logger.warning(
            "Async audit failed. Error: %s",
            exc,
        )
