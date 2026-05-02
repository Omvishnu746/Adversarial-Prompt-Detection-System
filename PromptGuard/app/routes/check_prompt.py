"""
PromptGuard – /check_prompt Route (Phase 3.5)

POST /check_prompt

Detection pipeline (Phase 3.5):
  Receive prompt
    ↓
  Preprocess
    ↓
  Rule Engine           (Tier 1 – BLOCK on regex match → skip remaining tiers)
    ↓
  Semantic Check        (Tier 2 – SBERT + FAISS)
    ↓
  DistilBERT Classifier (Tier 3 – sliding window chunking)
    ↓
  Risk Aggregation      (Phase 3.5 – weighted combination of all signals)
    ↓
  Final Decision        (BLOCK if final_risk_score > AGGREGATION_BLOCK_THRESHOLD)
    ↓
  Log (all signals + aggregation fields)
    ↓
  Respond

Backward compatibility:
  • If the semantic engine is not initialised, semantic_score = 0.0.
  • If the classifier is not loaded, classifier_score = chunk_risk_score = 0.0.
  • All tiers report their raw scores to the aggregator regardless of individual
    BLOCK decisions so the aggregated score is always consistent.
"""

from fastapi import APIRouter
from app.models.request_models import PromptRequest, PromptResponse
from app.models.semantic_response import SemanticResponse
from app.services.preprocessing import preprocess
from app.services.rule_engine import evaluate_rules
from app.services.semantic_engine import semantic_similarity_check, is_semantic_engine_ready
from app.services.classifier_engine import run_classifier
from app.services.model_loader import is_classifier_loaded
from app.services.aggregation_engine import aggregate_risk
from app.services.logger import log_request
from app.config.settings import (
    RULE_BLOCK_SCORE,
    RULE_ALLOW_SCORE,
    SEMANTIC_BLOCK_SCORE,
    CLASSIFIER_BLOCK_SCORE,
)

router = APIRouter()

# Risk threshold above which the aggregated score triggers a BLOCK.
# Kept lower than any single-tier BLOCK score so the aggregator can catch
# prompts that are moderately suspicious across multiple layers.
AGGREGATION_BLOCK_THRESHOLD: float = 0.65


@router.post(
    "/check_prompt",
    response_model=PromptResponse,
    summary="Evaluate a prompt for adversarial intent",
    description=(
        "Runs the Phase 3.5 four-tier detection pipeline: "
        "(1) Rule-based detection, "
        "(2) SBERT semantic similarity, "
        "(3) DistilBERT sequence classifier, "
        "(4) Risk Aggregation Engine. "
        "Returns a final risk score, routing decision, and the layer that triggered."
    ),
)
async def check_prompt(payload: PromptRequest) -> PromptResponse:
    """
    Phase 3.5 full detection pipeline:

    1. Preprocess the raw prompt
    2. Rule Engine [Tier 1]  — hard BLOCK on regex match (score = 1.0)
    3. Semantic Check [Tier 2] — collect similarity_score regardless of match
    4. DistilBERT Classifier [Tier 3] — collect adversarial_probability + chunk_risk_score
    5. Risk Aggregation [Phase 3.5] — weighted combination of all signals
       → final_risk_score > AGGREGATION_BLOCK_THRESHOLD → BLOCK, layer="aggregation"
    6. ALLOW if aggregation score is below threshold
    7. Log all signals + aggregation result
    8. Return PromptResponse with full breakdown
    """
    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    clean_prompt = preprocess(payload.prompt)

    # ── Step 2: Rule Engine (Tier 1) ─────────────────────────────────────────
    rule_result = evaluate_rules(clean_prompt)
    rule_score: float = 1.0 if rule_result["matched"] else 0.0

    # Hard BLOCK on rule match — skip remaining tiers entirely.
    if rule_result["matched"]:
        log_request(
            prompt=payload.prompt,
            risk_score=RULE_BLOCK_SCORE,
            decision="BLOCK",
            triggered_layer="rule",
            rule_score=rule_score,
            final_risk_score=RULE_BLOCK_SCORE,
        )
        return PromptResponse(
            risk_score=RULE_BLOCK_SCORE,
            decision="BLOCK",
            triggered_layer="rule",
            semantic_result=None,
            classifier_result=None,
            aggregation_result=None,
        )

    # ── Step 3: Semantic Check (Tier 2) ──────────────────────────────────────
    semantic_raw = semantic_similarity_check(clean_prompt)
    semantic_score: float = semantic_raw["similarity_score"]
    semantic_data = SemanticResponse(
        similarity_score=semantic_score,
        matched=semantic_raw["matched"],
        nearest_distance=semantic_raw["nearest_distance"],
    )

    # ── Step 4: DistilBERT Classifier (Tier 3) ───────────────────────────────
    classifier_data = None
    classifier_score: float = 0.0
    chunk_risk_score: float = 0.0

    if is_classifier_loaded():
        classifier_data = run_classifier(clean_prompt)
        classifier_score = classifier_data.adversarial_probability
        chunk_risk_score = classifier_data.chunk_risk_score

    # ── Step 5: Risk Aggregation (Phase 3.5) ─────────────────────────────────
    agg = aggregate_risk(
        rule_score=rule_score,
        semantic_score=semantic_score,
        classifier_score=classifier_score,
        chunk_risk_score=chunk_risk_score,
    )
    final_risk_score = agg["final_risk_score"]

    # ── Step 6: Final Decision ────────────────────────────────────────────────
    # Individual tier early-exits already handled rule (above).
    # For semantic / classifier, we now let the aggregator decide instead of
    # single-tier thresholds, giving a more nuanced combined signal.
    if semantic_raw["matched"]:
        # Semantic tier still hard-blocks by itself for high-confidence matches
        decision = "BLOCK"
        triggered_layer = "semantic"
        risk_score = SEMANTIC_BLOCK_SCORE
    elif classifier_data is not None and classifier_data.is_adversarial:
        # Classifier hard-blocks by itself when confidence exceeds ADVERSARIAL_THRESHOLD
        decision = "BLOCK"
        triggered_layer = "classifier"
        risk_score = CLASSIFIER_BLOCK_SCORE
    elif final_risk_score > AGGREGATION_BLOCK_THRESHOLD:
        # Aggregation catches prompts that are moderately suspicious across layers
        decision = "BLOCK"
        triggered_layer = "aggregation"
        risk_score = final_risk_score
    else:
        decision = "ALLOW"
        triggered_layer = "none"
        risk_score = final_risk_score

    # ── Step 7: Log ──────────────────────────────────────────────────────────
    log_request(
        prompt=payload.prompt,
        risk_score=risk_score,
        decision=decision,
        triggered_layer=triggered_layer,
        semantic_score=semantic_score,
        semantic_match=semantic_raw["matched"],
        classifier_score=classifier_score if classifier_data else None,
        classifier_triggered=classifier_data.is_adversarial if classifier_data else None,
        rule_score=rule_score,
        chunk_risk_score=chunk_risk_score if classifier_data else None,
        final_risk_score=final_risk_score,
    )

    # ── Step 8: Respond ──────────────────────────────────────────────────────
    return PromptResponse(
        risk_score=risk_score,
        decision=decision,
        triggered_layer=triggered_layer,
        semantic_result=semantic_data,
        classifier_result=classifier_data,
        aggregation_result=dict(agg),
    )
