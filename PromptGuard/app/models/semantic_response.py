"""
PromptGuard – Pydantic response model for Phase 2 Semantic Layer.

Defines SemanticResponse, which is embedded inside the updated PromptResponse
once Phase 2 is activated.
"""

from pydantic import BaseModel, Field


class SemanticResponse(BaseModel):
    """
    Structured output from the semantic similarity check.

    This model is embedded in the Phase 2 API response to expose the
    intermediate results of the SBERT + FAISS similarity check.

    Fields:
        similarity_score:  Cosine similarity between the input prompt and the
                           nearest known-attack embedding. Range [0.0, 1.0].
                           0.0 = no similarity / engine unavailable.
                           1.0 = identical to a stored attack pattern.

        matched:           True when similarity_score exceeds the configured
                           SEMANTIC_MATCH_THRESHOLD (default 0.90).  A True
                           value causes the pipeline to set decision = BLOCK
                           and triggered_layer = "semantic".

        nearest_distance:  Raw squared L2 distance returned by the FAISS index.
                           Useful for debugging and threshold calibration.
                           float('inf') when the semantic layer is unavailable
                           or the index is empty.
    """

    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity between the prompt and the nearest attack embedding "
            "in [0.0, 1.0].  0.0 when the semantic engine is not initialised."
        ),
        examples=[0.9432],
    )

    matched: bool = Field(
        ...,
        description=(
            "True if similarity_score exceeds the configured threshold (0.90). "
            "Triggers BLOCK decision with triggered_layer='semantic'."
        ),
        examples=[True],
    )

    nearest_distance: float = Field(
        ...,
        description=(
            "Raw squared L2 distance from the FAISS index search. "
            "Smaller values indicate greater similarity. "
            "Infinity signals an unavailable or empty index."
        ),
        examples=[0.1136],
    )
