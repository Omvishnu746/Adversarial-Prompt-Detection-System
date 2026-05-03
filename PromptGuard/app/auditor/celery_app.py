"""
PromptGuard – Celery Application (Phase 5)

Broker : Redis  (REDIS_URL)
Backend: Redis  (REDIS_URL)

Tasks are discovered automatically from app.auditor.audit_tasks.

Beat schedule (optional, requires celery beat):
  generate_daily_summary  runs every day at 00:00 UTC.
"""

from celery import Celery
from celery.schedules import crontab

from app.config.db_config import REDIS_URL

celery_app = Celery(
    "promptguard",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.auditor.audit_tasks"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task behaviour
    task_acks_late=True,           # acknowledge only after successful execution
    worker_prefetch_multiplier=1,  # one task at a time per worker process
    task_reject_on_worker_lost=True,
    # Result expiry (keep results 1 hour, then purge)
    result_expires=3600,
    # Beat schedule — daily summary at midnight UTC
    beat_schedule={
        "generate-daily-summary": {
            "task": "app.auditor.audit_tasks.generate_daily_summary",
            "schedule": crontab(hour=0, minute=0),
        },
    },
)
