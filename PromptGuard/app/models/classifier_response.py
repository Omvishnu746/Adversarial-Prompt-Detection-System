"""
PromptGuard – Pydantic response model for Phase 3 Classifier Layer.
"""

from pydantic import BaseModel, Field

class ClassifierResponse(BaseModel):
    """
    Structured output from the DistilBERT sequence classifier.
    
    Fields:
        is_adversarial: True if the model predicts the prompt is adversarial (class 0).
        adversarial_probability: Probability that the prompt is adversarial (0.0 to 1.0).
        benign_probability: Probability that the prompt is benign (0.0 to 1.0).
        max_chunk_index: If chunking was applied, the index of the chunk that yielded 
                         the highest adversarial probability.
    """
    is_adversarial: bool = Field(
        ...,
        description="True if the classifier predicts adversarial intent."
    )
    adversarial_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability [0.0, 1.0] that the prompt is adversarial."
    )
    benign_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability [0.0, 1.0] that the prompt is benign."
    )
    max_chunk_index: int = Field(
        default=0,
        ge=0,
        description="The index of the chunk that produced the highest adversarial probability."
    )
