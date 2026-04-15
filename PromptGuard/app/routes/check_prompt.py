"""
PromptGuard – /check_prompt Route

POST /check_prompt

Pipeline:
  Receive prompt → Preprocess → Rule engine → Assign risk score → Log → Respond
"""

from fastapi import APIRouter
from app.models.request_models import PromptRequest, PromptResponse
from app.services.preprocessing import preprocess
from app.services.rule_engine import evaluate_rules
from app.services.logger import log_request
from app.config.settings import RULE_BLOCK_SCORE, RULE_ALLOW_SCORE

router = APIRouter()


@router.post(
    "/check_prompt",
    response_model=PromptResponse,
    summary="Evaluate a prompt for adversarial intent",
    description=(
        "Runs the Phase 1 rule-based detection pipeline on the supplied prompt "
        "and returns a risk score, a routing decision, and the layer that triggered."
    ),
)
async def check_prompt(payload: PromptRequest) -> PromptResponse:
    """
    Phase 1 detection pipeline:

    1. Preprocess the raw prompt (unicode → lower → whitespace → base64 → leet)
    2. Evaluate against the rule engine
    3. Assign risk score:  1.0 if rule matched, else 0.0
    4. Log the full result as structured JSON
    5. Return PromptResponse
    """
    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    clean_prompt = preprocess(payload.prompt)

    # ── Step 2: Rule engine ──────────────────────────────────────────────────
    rule_result = evaluate_rules(clean_prompt)

    # ── Step 3: Risk score + decision ────────────────────────────────────────
    if rule_result["matched"]:
        risk_score = RULE_BLOCK_SCORE
        decision = "BLOCK"
        triggered_layer = "rule"
    else:
        risk_score = RULE_ALLOW_SCORE
        decision = "ALLOW"
        triggered_layer = "none"

    # ── Step 4: Log ──────────────────────────────────────────────────────────
    log_request(
        prompt=payload.prompt,
        risk_score=risk_score,
        decision=decision,
        triggered_layer=triggered_layer,
    )

    # ── Step 5: Respond ──────────────────────────────────────────────────────
    return PromptResponse(
        risk_score=risk_score,
        decision=decision,
        triggered_layer=triggered_layer,
    )
