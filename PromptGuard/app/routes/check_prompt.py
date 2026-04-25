"""
PromptGuard – /check_prompt Route (Phase 2)

POST /check_prompt

Detection pipeline (Phase 2):
  Receive prompt
    ↓
  Preprocess
    ↓
  Rule Engine         (Phase 1 – BLOCK on regex match → skip semantic)
    ↓
  Semantic Check      (Phase 2 – SBERT + FAISS, runs only when engine ready)
    ↓
  Assign Risk Score + Decision
    ↓
  Log (with semantic fields)
    ↓
  Respond

Backward compatibility:
  • If the semantic engine is not yet initialised (model/index not loaded),
    the route behaves identically to Phase 1.
  • The semantic_result field in the response is None when the engine is off.
"""

from fastapi import APIRouter
from app.models.request_models import PromptRequest, PromptResponse
from app.models.semantic_response import SemanticResponse
from app.services.preprocessing import preprocess
from app.services.rule_engine import evaluate_rules
from app.services.semantic_engine import (
    semantic_similarity_check,
    is_semantic_engine_ready,
    SEMANTIC_MATCH_THRESHOLD,
)
from app.services.logger import log_request
from app.config.settings import (
    RULE_BLOCK_SCORE,
    RULE_ALLOW_SCORE,
    SEMANTIC_BLOCK_SCORE,
)

router = APIRouter()


@router.post(
    "/check_prompt",
    response_model=PromptResponse,
    summary="Evaluate a prompt for adversarial intent",
    description=(
        "Runs the Phase 1 rule-based detection pipeline followed by the Phase 2 "
        "SBERT semantic similarity check (when the engine is initialised) and "
        "returns a risk score, a routing decision, and the layer that triggered."
    ),
)
async def check_prompt(payload: PromptRequest) -> PromptResponse:
    """
    Phase 1 + Phase 2 detection pipeline:

    1. Preprocess the raw prompt (unicode → lower → whitespace → base64 → leet)
    2. Evaluate against the rule engine  [Phase 1]
       → If matched: risk=1.0, BLOCK, layer="rule" (skip semantic check)
    3. Run semantic similarity check     [Phase 2, if engine ready]
       → If similarity_score > 0.90: risk=0.9, BLOCK, layer="semantic"
    4. Assign risk score + decision
    5. Log the full result (with semantic fields)
    6. Return PromptResponse
    """
    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    clean_prompt = preprocess(payload.prompt)

    # ── Step 2: Rule engine (Phase 1) ────────────────────────────────────────
    rule_result = evaluate_rules(clean_prompt)

    if rule_result["matched"]:
        # Rule matched → immediate BLOCK, skip semantic layer
        risk_score = RULE_BLOCK_SCORE          # 1.0
        decision = "BLOCK"
        triggered_layer = "rule"
        semantic_data = None

        log_request(
            prompt=payload.prompt,
            risk_score=risk_score,
            decision=decision,
            triggered_layer=triggered_layer,
            semantic_score=None,
            semantic_match=None,
        )

        return PromptResponse(
            risk_score=risk_score,
            decision=decision,
            triggered_layer=triggered_layer,
            semantic_result=None,
        )

    # ── Step 3: Semantic Similarity Check (Phase 2) ───────────────────────────
    semantic_result_raw = semantic_similarity_check(clean_prompt)

    semantic_data: SemanticResponse | None = SemanticResponse(
        similarity_score=semantic_result_raw["similarity_score"],
        matched=semantic_result_raw["matched"],
        nearest_distance=semantic_result_raw["nearest_distance"],
    )

    if semantic_result_raw["matched"]:
        # Semantic match → BLOCK with degraded risk score (0.9 < rule's 1.0)
        risk_score = SEMANTIC_BLOCK_SCORE      # 0.9
        decision = "BLOCK"
        triggered_layer = "semantic"
    else:
        # No threat detected by either layer
        risk_score = RULE_ALLOW_SCORE          # 0.0
        decision = "ALLOW"
        triggered_layer = "none"

    # ── Step 4: Log ──────────────────────────────────────────────────────────
    log_request(
        prompt=payload.prompt,
        risk_score=risk_score,
        decision=decision,
        triggered_layer=triggered_layer,
        semantic_score=semantic_result_raw["similarity_score"],
        semantic_match=semantic_result_raw["matched"],
    )

    # ── Step 5: Respond ──────────────────────────────────────────────────────
    return PromptResponse(
        risk_score=risk_score,
        decision=decision,
        triggered_layer=triggered_layer,
        semantic_result=semantic_data,
    )
