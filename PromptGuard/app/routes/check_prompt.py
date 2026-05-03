"""
PromptGuard – /check_prompt Route (Phase 4 - Final)

Detection pipeline:

  1. Preprocess
  2. Rule Engine        → HARD BLOCK if matched
  3. Semantic Check     → HARD BLOCK if matched (similarity >= threshold)
  4. DistilBERT         → HARD BLOCK if is_adversarial=True
  5. Aggregation
       • RAW  aggregation  — raw scores, used for risk_score display
       • BINARY aggregation — binarized 0/1, used for grey-zone routing
  6. Router (Tier 4)    → ALLOW / SANITIZE for grey zone only
  7. Log + Respond

Transparency design
────────────────────
risk_score  = raw final_risk_score  — always reflects actual signal strength
              For hard blocks this will be naturally high (classifier + semantic).
              For ALLOW this will be naturally low.

confidence  = same value (router_result.confidence = raw final_risk_score)

final_risk_score in aggregation_result = raw aggregation, so users see the
realistic combined threat level from all layers.

Binarized scores are only used as the internal router input to avoid
contaminated raw probabilities routing benign phrases to SANITIZE.
"""

from fastapi import APIRouter

from app.models.request_models import PromptRequest, PromptResponse
from app.models.semantic_response import SemanticResponse
from app.models.router_response import RouterResponse
from app.services.preprocessing import preprocess
from app.services.rule_engine import evaluate_rules
from app.services.semantic_engine import semantic_similarity_check
from app.services.classifier_engine import run_classifier
from app.services.model_loader import is_classifier_loaded
from app.services.aggregation_engine import aggregate_risk
from app.services.router_engine import route_decision
from app.services.logger import log_request
from app.config.router_config import CHUNK_BLOCK_THRESHOLD
from app.auditor.audit_logger import trigger_audit_log

router = APIRouter()


@router.post(
    "/check_prompt",
    response_model=PromptResponse,
    summary="Evaluate a prompt for adversarial intent",
    description=(
        "Full Phase 4 detection pipeline: Rule → Semantic → DistilBERT → "
        "Aggregation → Decision Router (ALLOW / BLOCK / SANITIZE)."
    ),
)
async def check_prompt(payload: PromptRequest) -> PromptResponse:

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    clean_prompt = preprocess(payload.prompt)

    # ── 2. Rule Engine ────────────────────────────────────────────────────────
    rule_result  = evaluate_rules(clean_prompt)
    rule_score: float = 1.0 if rule_result["matched"] else 0.0

    # ── 3. Semantic Check ─────────────────────────────────────────────────────
    semantic_raw  = semantic_similarity_check(clean_prompt)
    semantic_score: float = semantic_raw["similarity_score"]
    semantic_data = SemanticResponse(
        similarity_score=semantic_score,
        matched=semantic_raw["matched"],
        nearest_distance=semantic_raw["nearest_distance"],
    )

    # ── 4. DistilBERT Classifier ──────────────────────────────────────────────
    classifier_data    = None
    raw_cls_score: float = 0.0
    chunk_risk_score: float = 0.0

    if is_classifier_loaded():
        classifier_data = run_classifier(clean_prompt)
        raw_cls_score   = classifier_data.adversarial_probability
        chunk_risk_score = classifier_data.chunk_risk_score

    # ── 5. Aggregation (RAW) ──────────────────────────────────────────────────
    # Always compute with raw scores so final_risk_score honestly reflects all
    # layer signals. This is what gets shown in the response.
    agg_raw = aggregate_risk(
        rule_score=rule_score,
        semantic_score=semantic_score,
        classifier_score=raw_cls_score,
        chunk_risk_score=chunk_risk_score,
    )
    raw_final_risk: float = agg_raw["final_risk_score"]

    # ── 5b. Aggregation (BINARY) ──────────────────────────────────────────────
    # Used only as input to the grey-zone router to prevent contaminated raw
    # probabilities from routing benign phrases to SANITIZE.
    cls_bin: float   = 1.0 if (classifier_data and classifier_data.is_adversarial) else 0.0
    chunk_bin: float = 1.0 if chunk_risk_score > CHUNK_BLOCK_THRESHOLD else 0.0

    agg_bin = aggregate_risk(
        rule_score=rule_score,
        semantic_score=semantic_score,
        classifier_score=cls_bin,
        chunk_risk_score=chunk_bin,
    )
    bin_final_risk: float = agg_bin["final_risk_score"]

    # ── 6. Hard-block check ───────────────────────────────────────────────────
    # Any high-confidence individual signal triggers an immediate BLOCK.
    # The router (grey-zone) is only used when none of these fire.
    hard_block = False
    triggered_layer = "none"

    if rule_result["matched"]:
        hard_block      = True
        triggered_layer = "rule"

    elif semantic_raw["matched"]:
        hard_block      = True
        triggered_layer = "semantic"

    elif classifier_data is not None and classifier_data.is_adversarial:
        hard_block      = True
        triggered_layer = "classifier"

    # ── 7. Final Decision ─────────────────────────────────────────────────────
    if hard_block:
        # Use the RAW aggregated score as the reported risk/confidence —
        # this naturally reflects all the signals (e.g. 0.62 when semantic
        # and classifier are both high), making the numbers intuitive.
        decision   = "BLOCK"
        risk_score = raw_final_risk
        router_out = RouterResponse(
            decision="BLOCK",
            confidence=round(raw_final_risk, 4),
            reason=f"{triggered_layer.capitalize()}-based block — "
                   f"raw risk {raw_final_risk:.4f}",
            sanitized_text=None,
        )
    else:
        # Grey zone — let the router decide using binarized aggregation
        router_out = route_decision(
            final_risk_score=bin_final_risk,
            max_chunk_score=chunk_risk_score,
            rule_score=rule_score,
            original_prompt=payload.prompt,
        )
        decision       = router_out.decision
        risk_score     = router_out.confidence
        triggered_layer = "none" if decision == "ALLOW" else "aggregation"

    # ── 8. Log (structured file logger) ───────────────────────────────────────
    log_request(
        prompt=payload.prompt,
        risk_score=round(risk_score, 4),
        decision=decision,
        triggered_layer=triggered_layer,
        semantic_score=semantic_score,
        semantic_match=semantic_raw["matched"],
        classifier_score=raw_cls_score if classifier_data else None,
        classifier_triggered=classifier_data.is_adversarial if classifier_data else None,
        rule_score=rule_score,
        chunk_risk_score=chunk_risk_score if classifier_data else None,
        final_risk_score=raw_final_risk,
        router_decision=decision,
        router_reason=router_out.reason,
        sanitized=decision == "SANITIZE",
    )

    # ── 9. Async Audit (Phase 5) ─────────────────────────────────────────────
    # Fire-and-forget: dispatched to Celery worker via Redis.
    # Returns in < 1 ms — does NOT block the API response.
    trigger_audit_log(
        prompt=payload.prompt,
        decision=decision,
        triggered_layer=triggered_layer,
        rule_score=rule_score,
        semantic_score=semantic_score,
        classifier_score=raw_cls_score,
        chunk_risk_score=chunk_risk_score,
        final_risk_score=raw_final_risk,
        sanitized=decision == "SANITIZE",
    )

    # ── 9. Respond ────────────────────────────────────────────────────────────
    return PromptResponse(
        risk_score=round(risk_score, 4),
        decision=decision,
        triggered_layer=triggered_layer,
        semantic_result=semantic_data,
        classifier_result=classifier_data,
        aggregation_result={
            **agg_raw,                         # raw scores — honest signal picture
            "binary_final_risk_score": round(bin_final_risk, 4),  # grey-zone router input
        },
        router_result=router_out,
    )
