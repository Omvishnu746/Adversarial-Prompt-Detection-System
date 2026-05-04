"""
PromptGuard – Celery Audit Tasks (Phase 5)

Tasks
─────
log_attack_event        Insert one AttackLog row per /check_prompt call.
generate_daily_summary  Aggregate daily stats into DailySummary (beat-scheduled).

Both tasks run in a Celery worker process — completely off the API request path,
so they add zero latency to HTTP responses.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker

from app.auditor.db_models import AttackLog, Base, DailySummary
from app.config.db_config import (
    AUTO_UPDATE_THREAT_INDEX,
    DATABASE_URL,
    EXPLAINABILITY_TOP_K,
    MAX_STORED_PROMPT_LENGTH,
)

logger = logging.getLogger("promptguard.audit_tasks")

# ── SQLAlchemy engine (created once per worker process) ──────────────────────
_engine = None
_Session = None


def _get_session():
    """Lazily create the DB engine and return a session."""
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        Base.metadata.create_all(_engine)          # idempotent: creates tables if missing
        _Session = sessionmaker(bind=_engine)
    return _Session()


# ── Task 1: log_attack_event ─────────────────────────────────────────────────

def log_attack_event(
    *,
    prompt: str,
    decision: str,
    triggered_layer: str = "none",
    rule_score: float = 0.0,
    semantic_score: float = 0.0,
    classifier_score: float = 0.0,
    chunk_risk_score: float = 0.0,
    final_risk_score: float = 0.0,
    sanitized: bool = False,
    target_llm: str | None = None,
    timestamp: str | None = None,
) -> dict:
    """
    Insert one audit row into attack_logs.

    Also:
      • Runs explainability to find top suspicious tokens (BLOCK only).
      • Optionally adds the prompt embedding to the FAISS threat index (BLOCK only).
    """
    try:
        # ── Explainability (only for blocked prompts — highest value there) ──
        top_tokens_json: str | None = None
        if decision == "BLOCK":
            try:
                from app.auditor.explainability import get_top_tokens, tokens_to_json
                tokens = get_top_tokens(
                    prompt[:MAX_STORED_PROMPT_LENGTH],
                    top_k=EXPLAINABILITY_TOP_K,
                )
                top_tokens_json = tokens_to_json(tokens) if tokens else None
            except Exception as ex:
                logger.warning("Explainability skipped: %s", ex)

        # ── Threat index update (FAISS) ──────────────────────────────────────
        threat_stored = False
        if decision == "BLOCK" and AUTO_UPDATE_THREAT_INDEX:
            try:
                from app.auditor.threat_memory import add_to_threat_index
                threat_stored = add_to_threat_index(prompt)
            except Exception as ex:
                logger.warning("Threat index update skipped: %s", ex)

        # ── Build timestamp ───────────────────────────────────────────────────
        ts = (
            datetime.fromisoformat(timestamp)
            if timestamp
            else datetime.now(timezone.utc)
        )

        # ── Write to database ─────────────────────────────────────────────────
        session = _get_session()
        try:
            log = AttackLog(
                prompt_text=prompt[:MAX_STORED_PROMPT_LENGTH],
                decision=decision,
                triggered_layer=triggered_layer,
                rule_score=rule_score,
                semantic_score=semantic_score,
                classifier_score=classifier_score,
                chunk_risk_score=chunk_risk_score,
                final_risk_score=final_risk_score,
                sanitized_flag=sanitized,
                target_llm=target_llm,
                top_tokens=top_tokens_json,
                threat_stored=threat_stored,
                timestamp=ts,
            )
            session.add(log)
            session.commit()
            logger.info(
                "Audit logged | decision=%s layer=%s risk=%.4f id=%s",
                decision, triggered_layer, final_risk_score, log.id,
            )
            return {"status": "ok", "id": log.id}

        except Exception as db_exc:
            session.rollback()
            raise db_exc
        finally:
            session.close()

    except Exception as exc:
        logger.error("log_attack_event failed: %s", exc)
        raise exc


# ── Task 2: generate_daily_summary ───────────────────────────────────────────

def generate_daily_summary(target_date: str | None = None) -> dict:
    """
    Aggregate attack_logs for a given date into daily_summaries.

    Parameters
    ----------
    target_date : str or None
        ISO date string YYYY-MM-DD.  Defaults to today (UTC).
    """
    try:
        today = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        session = _get_session()
        try:
            # Aggregate using SQL
            rows = (
                session.query(
                    AttackLog.decision,
                    func.count(AttackLog.id).label("cnt"),
                )
                .filter(func.date(AttackLog.timestamp) == today)
                .group_by(AttackLog.decision)
                .all()
            )

            counts = {row.decision: row.cnt for row in rows}
            blocked   = counts.get("BLOCK",    0)
            sanitized = counts.get("SANITIZE", 0)
            allowed   = counts.get("ALLOW",    0)
            total     = blocked + sanitized + allowed

            # Upsert into daily_summaries
            existing = (
                session.query(DailySummary)
                .filter(DailySummary.date == today)
                .first()
            )
            if existing:
                existing.total_requests  = total
                existing.blocked_count   = blocked
                existing.sanitized_count = sanitized
                existing.allowed_count   = allowed
                existing.updated_at      = datetime.now(timezone.utc)
            else:
                session.add(DailySummary(
                    date=today,
                    total_requests=total,
                    blocked_count=blocked,
                    sanitized_count=sanitized,
                    allowed_count=allowed,
                    updated_at=datetime.now(timezone.utc),
                ))

            session.commit()
            logger.info(
                "Daily summary [%s]: total=%d block=%d sanitize=%d allow=%d",
                today, total, blocked, sanitized, allowed,
            )
            return {
                "date": today,
                "total": total,
                "blocked": blocked,
                "sanitized": sanitized,
                "allowed": allowed,
            }

        finally:
            session.close()

    except Exception as exc:
        logger.error("generate_daily_summary failed: %s", exc)
        raise exc
