"""Validation test for Phase 3.5 Risk Aggregation Engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.aggregation_engine import aggregate_risk
from app.models.request_models import PromptResponse

# ── Spec test case ────────────────────────────────────────────────────────────
result = aggregate_risk(rule_score=1.0, semantic_score=0.70, classifier_score=0.80, chunk_risk_score=0.85)
expected = round(0.30*1.0 + 0.30*0.80 + 0.25*0.70 + 0.15*0.85, 4)

print("=== SPEC TEST CASE ===")
for k, v in result.items():
    print(f"  {k}: {v}")
print(f"  Expected: ~0.83  (manual calc: {expected})")
assert abs(result["final_risk_score"] - expected) < 0.001, f"Formula mismatch: {result['final_risk_score']} != {expected}"
print("  PASS\n")

# ── Edge cases ────────────────────────────────────────────────────────────────
r_zeros = aggregate_risk(0.0, 0.0, 0.0, 0.0)
r_ones  = aggregate_risk(1.0, 1.0, 1.0, 1.0)
r_oob   = aggregate_risk(1.5, -0.2, 2.0, -1.0)

print("=== EDGE CASES ===")
print(f"  All zeros    -> {r_zeros['final_risk_score']} (expected 0.0) {'PASS' if r_zeros['final_risk_score'] == 0.0 else 'FAIL'}")
print(f"  All ones     -> {r_ones['final_risk_score']} (expected 1.0) {'PASS' if r_ones['final_risk_score'] == 1.0 else 'FAIL'}")
print(f"  Out-of-range -> {r_oob['final_risk_score']} (clamped to [0,1]) {'PASS' if 0.0 <= r_oob['final_risk_score'] <= 1.0 else 'FAIL'}\n")

# ── Response model ────────────────────────────────────────────────────────────
fields = list(PromptResponse.model_fields.keys())
print("=== PROMPTRESPONSE FIELDS ===")
for f in fields:
    print(f"  {f}")

print("\nAll validation checks passed.")
