"""
PromptGuard – Threat Memory (Phase 5)

Stores embeddings of blocked prompts so the FAISS semantic index grows
smarter over time without retraining the SBERT model.

Design
──────
• In-process: Blocked prompt embeddings are added to the live FAISS index in
  the API process via an HTTP call to an internal admin endpoint, OR saved to a
  pending-threats file and picked up on next server restart.

• Worker-process: The Celery worker cannot directly mutate the API process's
  FAISS index (different memory space). Instead it writes the embedding +
  prompt to PostgreSQL (threat_embeddings table) and flags it as pending.

• On API startup: app/main.py calls load_pending_threats() to pull any new
  embeddings from the DB into the live FAISS index.

This approach is safe for CPU-only, single-server deployments.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("promptguard.threat_memory")


def add_to_threat_index(prompt: str) -> bool:
    """
    Encode *prompt* and save its embedding to the pending-threats DB table.
    Returns True on success, False otherwise.

    The embedding will be loaded into the live FAISS index on the next
    server restart (or via POST /admin/reload_threats).
    """
    try:
        from sqlalchemy import Column, DateTime, Integer, LargeBinary, Text, create_engine
        from sqlalchemy.orm import declarative_base, sessionmaker
        from datetime import datetime, timezone
        from app.config.db_config import DATABASE_URL

        # ── Encode with SBERT ─────────────────────────────────────────────────
        from sentence_transformers import SentenceTransformer
        import os

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "sbert_index", "model"
        )
        # Fall back to the hub name if local path is missing
        if not os.path.isdir(model_path):
            model_path = "all-MiniLM-L6-v2"

        sbert = SentenceTransformer(model_path)
        embedding: np.ndarray = sbert.encode([prompt], normalize_embeddings=True)[0]

        # ── Persist to pending_threats table ──────────────────────────────────
        Base = declarative_base()

        class PendingThreat(Base):
            __tablename__ = "pending_threats"
            id        = Column(Integer, primary_key=True)
            prompt    = Column(Text)
            embedding = Column(LargeBinary)            # raw float32 bytes
            created   = Column(DateTime(timezone=True))

        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            session.add(PendingThreat(
                prompt=prompt[:500],
                embedding=embedding.astype(np.float32).tobytes(),
                created=datetime.now(timezone.utc),
            ))
            session.commit()
            logger.info("Threat embedding saved for future FAISS update.")
            return True
        finally:
            session.close()

    except Exception as exc:
        logger.warning("add_to_threat_index failed: %s", exc)
        return False


def load_pending_threats() -> int:
    """
    Pull all pending threat embeddings from the DB into the live FAISS index.
    Called once during server startup (app/main.py).

    Returns the number of new threats loaded.
    """
    loaded = 0
    try:
        from sqlalchemy import Column, DateTime, Integer, LargeBinary, Text, create_engine
        from sqlalchemy.orm import declarative_base, sessionmaker
        from app.config.db_config import DATABASE_URL
        import faiss

        Base = declarative_base()

        class PendingThreat(Base):
            __tablename__ = "pending_threats"
            id        = Column(Integer, primary_key=True)
            prompt    = Column(Text)
            embedding = Column(LargeBinary)
            created   = Column(DateTime(timezone=True))

        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            rows = session.query(PendingThreat).all()
            if not rows:
                return 0

            # Load the live FAISS index
            from app.services.semantic_engine import get_faiss_index
            index = get_faiss_index()
            if index is None:
                return 0

            for row in rows:
                vec = np.frombuffer(row.embedding, dtype=np.float32).reshape(1, -1)
                index.add(vec)
                loaded += 1

            # Clear the pending table
            session.query(PendingThreat).delete()
            session.commit()
            logger.info("Loaded %d pending threats into FAISS index.", loaded)

        finally:
            session.close()

    except Exception as exc:
        logger.warning("load_pending_threats failed: %s", exc)

    return loaded
