"""
PromptGuard – Pydantic models for user-facing explainability.

These models represent the abstracted, public-safe explanation payload
that is attached to every API response.  They deliberately omit any
internal detail (scores, thresholds, regex patterns, model internals).
"""

from typing import List
from pydantic import BaseModel, Field


class ExplanationReason(BaseModel):
    """A single, abstracted reason for the detection decision."""

    category: str = Field(
        ...,
        description="High-level category name of the detected behaviour (e.g. 'Instruction Override').",
    )
    message: str = Field(
        ...,
        description="User-facing description of the category — never exposes internal detection logic.",
    )


class ExplanationResponse(BaseModel):
    """
    Structured, user-facing explainability payload.

    Attached to every PromptResponse so the caller understands *why* a
    decision was made, expressed at a safe level of abstraction.

    Fields:
        reasons     – One entry per unique detection category that fired.
                      Empty list when the prompt was ALLOWED without flags.
        suggestions – Up to 2 actionable, safe-to-expose tips for the user.
                      Empty list when no actionable advice applies.
    """

    reasons: List[ExplanationReason] = Field(
        default_factory=list,
        description="List of abstracted reasons that contributed to the decision.",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Up to 2 user-safe suggestions on how to rephrase the prompt.",
    )
