"""
PromptGuard – Pydantic response model for Phase 4 Router Layer.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class RouterResponse(BaseModel):
    """
    Structured output from the Phase 4 Decision Router (Tier 4).

    Fields:
        decision:        Final enforcement action — ALLOW, BLOCK, or SANITIZE.
        confidence:      The final_risk_score that drove the decision (0.0–1.0).
        risk_level:      Human-readable risk band — LOW, MEDIUM, or HIGH.
                         Derived from the final_risk_score; never exposes the
                         raw numeric value in the public API.
        reason:          Human-readable explanation of why this decision was made.
        sanitized_text:  Cleaned prompt text when decision == SANITIZE; None otherwise.
    """

    decision: Literal["ALLOW", "BLOCK", "SANITIZE"] = Field(
        ...,
        description="Final enforcement action produced by the Tier 4 router.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final risk score that drove the routing decision (0.0–1.0).",
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ...,
        description="Abstracted risk band: LOW, MEDIUM, or HIGH.",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of the routing decision.",
    )
    sanitized_text: Optional[str] = Field(
        default=None,
        description=(
            "Cleaned prompt text with adversarial patterns removed. "
            "Only present when decision == 'SANITIZE'."
        ),
    )
