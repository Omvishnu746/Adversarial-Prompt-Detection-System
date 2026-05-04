"""
PromptGuard – Pydantic response model for Phase 3 Classifier Layer.
"""

from typing import List

from pydantic import BaseModel, Field

class ClassifierResponse(BaseModel):
    """
    Structured output from the DistilBERT sequence classifier.

    Fields:
        is_adversarial:        True if the model predicts adversarial intent.
        adversarial_probability: Max adversarial probability across all chunks (0.0–1.0).
        benign_probability:    Benign probability of the most adversarial chunk (0.0–1.0).
        max_chunk_index:       Index of the chunk that yielded the highest adversarial probability.
        chunk_risk_score:      Alias of adversarial_probability — the maximum adversarial
                               probability across all chunks, used by the aggregation engine.
    """
    is_adversarial: bool = Field(
        ...,
        description="True if the classifier predicts adversarial intent."
    )
    adversarial_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Max adversarial probability [0.0, 1.0] across all chunks."
    )
    benign_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Benign probability [0.0, 1.0] of the most adversarial chunk."
    )
    max_chunk_index: int = Field(
        default=0,
        ge=0,
        description="Index of the chunk that produced the highest adversarial probability."
    )
    chunk_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum adversarial probability across all chunks (0.0–1.0). "
            "Equal to adversarial_probability for single-chunk prompts. "
            "Used directly by the Phase 3.5 Risk Aggregation Engine."
        ),
    )
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "Safe, abstracted category tags emitted for user-facing explainability. "
            "Contains ['classifier_high_risk'] when is_adversarial=True, "
            "otherwise an empty list. Never exposes probabilities or thresholds."
        ),
    )
