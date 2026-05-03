# PromptGuard – Phase 5 Setup Guide

## Prerequisites
- Docker Desktop installed and running
- Python environment with all dependencies

---

## Step 1 — Start Infrastructure (Redis + PostgreSQL)

```powershell
# From the promptguard/ directory
docker compose up -d
```

Verify both containers are healthy:
```powershell
docker compose ps
```

---

## Step 2 — Install Phase 5 Python Dependencies

```powershell
pip install "celery[redis]>=5.3.0" "redis>=5.0.0" "sqlalchemy>=2.0.0" "psycopg2-binary>=2.9.0"
```

---

## Step 3 — Start the Celery Worker (new terminal)

```powershell
# From the promptguard/ directory
python -m celery -A app.auditor.celery_app worker --loglevel=info --concurrency=2
```

---

## Step 4 — (Optional) Start Celery Beat for Daily Summaries

```powershell
python -m celery -A app.auditor.celery_app beat --loglevel=info
```

---

## Step 5 — Start the API (existing terminal)

```powershell
python run.py
```

---

## Verification

Send an adversarial prompt, then query the database:

```powershell
# Check that the audit row was inserted
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('postgresql://promptguard:veritext123@localhost:5432/promptguard')
with engine.connect() as conn:
    rows = conn.execute(text('SELECT id, decision, triggered_layer, final_risk_score, timestamp FROM attack_logs ORDER BY id DESC LIMIT 5'))
    for r in rows:
        print(r)
"
```

---

## Environment Variables (.env)

```
DATABASE_URL=postgresql://promptguard:veritext123@localhost:5432/promptguard
REDIS_URL=redis://localhost:6379/0
MAX_STORED_PROMPT_LENGTH=1000
EXPLAINABILITY_TOP_K=10
AUTO_UPDATE_THREAT_INDEX=true
```

---

## Architecture

```
HTTP Request
    │
    ▼
FastAPI (check_prompt)
    │
    ├── Rule Engine
    ├── Semantic Engine
    ├── DistilBERT Classifier
    ├── Aggregation
    ├── Router Decision  ← returns to client immediately
    │
    └── trigger_audit_log()   ← fire-and-forget (< 1 ms)
                │
                ▼
            Redis Queue
                │
                ▼
         Celery Worker
                │
        ┌───────┴────────┐
        │                │
   Explainability    PostgreSQL
   (top tokens)    (attack_logs)
        │
   FAISS Update
  (pending_threats)
```

---

## API remains fast even without auditor

If Redis is not running, `trigger_audit_log()` catches the exception,
logs a warning once, and silently disables itself. The API continues
to function normally — audit logging is non-blocking and optional.
