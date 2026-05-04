"""
PromptGuard – Database & Redis Configuration (Phase 5)

Environment variables (set in .env):
  DATABASE_URL  PostgreSQL connection string
  REDIS_URL     Redis connection string
"""

import os

# ── PostgreSQL ────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///promptguard.db",
)

# ── Redis (Celery broker + result backend) ────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Auditor settings ──────────────────────────────────────────────────────────
# Maximum prompt length stored in the database
MAX_STORED_PROMPT_LENGTH: int = int(os.getenv("MAX_STORED_PROMPT_LENGTH", "1000"))

# Top-N tokens returned by the explainability module
EXPLAINABILITY_TOP_K: int = int(os.getenv("EXPLAINABILITY_TOP_K", "10"))

# Whether to auto-add blocked prompts to FAISS threat index
AUTO_UPDATE_THREAT_INDEX: bool = os.getenv("AUTO_UPDATE_THREAT_INDEX", "true").lower() == "true"
