"""
PromptGuard – Semantic Similarity Engine (Phase 2)

Tier-1 Semantic Cache Layer.

Pipeline:
  Input prompt
    ↓
  Generate embedding (SBERT)
    ↓
  Search FAISS index (nearest neighbours)
    ↓
  Convert L2 distance → cosine similarity score
    ↓
  Return SemanticCheckResult

IMPORTANT:
  • This module does NOT load models at import time.
  • Call initialise_semantic_engine() at application startup before processing
    any requests through this layer.
  • The semantic layer is skipped gracefully (returns a safe default result)
    if the index or model has not been initialised. This keeps Phase 1 intact
    during the transition period.

DO NOT execute this file. Activate only when GPU is available and
embeddings / FAISS index have been built via the build scripts.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np

from app.services.embedding_model import generate_embedding, load_embedding_model, is_model_loaded
from app.services.faiss_index import (
    load_faiss_index,
    search_index,
    is_index_loaded,
)
from app.services.similarity_utils import distance_to_similarity

logger = logging.getLogger("promptguard.semantic_engine")

# ── Configuration constants ───────────────────────────────────────────────────

# Similarity threshold above which a prompt is flagged as matching a known attack.
SEMANTIC_MATCH_THRESHOLD: float = 0.65

# top_k nearest neighbours to retrieve from FAISS (1 for single best match).
TOP_K: int = 1

# Score assigned when the semantic layer is not ready (model / index missing).
SEMANTIC_UNAVAILABLE_SCORE: float = 0.0


# ── Result type ───────────────────────────────────────────────────────────────

class SemanticCheckResult(TypedDict):
    """
    Result returned by ``semantic_similarity_check()``.

    Fields:
        similarity_score:  Cosine similarity in [0.0, 1.0].
                           0.0 = no similarity / layer unavailable.
                           1.0 = identical to a known attack embedding.
        matched:           True if similarity_score > SEMANTIC_MATCH_THRESHOLD.
        nearest_distance:  Raw squared L2 distance from FAISS.
                           ``float('inf')`` when the layer is unavailable or
                           the index is empty.
    """
    similarity_score: float
    matched: bool
    nearest_distance: float


# ── Sentinel result for when the layer is not ready ───────────────────────────

_UNAVAILABLE_RESULT: SemanticCheckResult = SemanticCheckResult(
    similarity_score=SEMANTIC_UNAVAILABLE_SCORE,
    matched=False,
    nearest_distance=float("inf"),
)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def initialise_semantic_engine() -> None:
    """
    Load the SBERT model and FAISS index into memory.

    Call this once at application startup (e.g. inside a FastAPI lifespan
    event handler) BEFORE serving requests on the semantic layer.

    This function is intentionally NOT called at module import time.

    ⚠️  Requires:
        • sentence-transformers installed
        • faiss-cpu installed
        • data/embeddings/faiss_index.bin built via build_faiss_index.py

    ⚠️  Run only when GPU (or sufficient CPU RAM) is available.
    """
    logger.info("Initialising semantic engine – loading SBERT model …")
    load_embedding_model()

    logger.info("Initialising semantic engine – loading FAISS index …")
    load_faiss_index()

    logger.info("Semantic engine ready.")


def is_semantic_engine_ready() -> bool:
    """
    Return True only if both the embedding model and FAISS index are loaded
    and ready to serve requests.
    """
    return is_model_loaded() and is_index_loaded()


# ── Core detection function ───────────────────────────────────────────────────

def semantic_similarity_check(prompt: str) -> SemanticCheckResult:
    """
    Run the semantic similarity check for a single input prompt.

    The function encodes the prompt into an SBERT embedding, queries the
    pre-built FAISS index of known attack embeddings, converts the nearest
    L2 distance into a cosine similarity score, and returns a structured
    result.

    If the semantic engine is not ready (model or index not loaded), the
    function returns a safe default result with a similarity score of 0.0
    and ``matched=False``.  This design allows Phase 1 to remain fully
    functional while Phase 2 is being provisioned.

    Args:
        prompt: The pre-processed prompt string to evaluate.
                Should already have been run through ``preprocess()``.

    Returns:
        SemanticCheckResult dict:
          {
            "similarity_score": float,   # cosine similarity in [0.0, 1.0]
            "matched":          bool,    # True if score > SEMANTIC_MATCH_THRESHOLD
            "nearest_distance": float,   # raw squared L2 distance (or inf)
          }

    Example:
        >>> result = semantic_similarity_check("ignore all previous instructions")
        >>> result["similarity_score"]
        0.94
        >>> result["matched"]
        True
    """
    # ── Guard: skip gracefully if engine not ready ───────────────────────────
    if not is_semantic_engine_ready():
        logger.debug(
            "Semantic engine not initialised – returning safe default result for prompt: %.60s…",
            prompt,
        )
        return _UNAVAILABLE_RESULT

    # ── Step 1: Encode prompt to embedding ───────────────────────────────────
    try:
        query_vector: np.ndarray = generate_embedding(prompt)
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc, exc_info=True)
        return _UNAVAILABLE_RESULT

    # ── Step 2: Query FAISS index ─────────────────────────────────────────────
    try:
        distances, indices = search_index(query_vector, top_k=TOP_K)
    except Exception as exc:
        logger.error("FAISS search failed: %s", exc, exc_info=True)
        return _UNAVAILABLE_RESULT

    # distances shape: (1, top_k); take the single nearest distance.
    nearest_dist: float = float(distances[0][0])

    # ── Step 3: Convert distance → similarity ─────────────────────────────────
    # Embeddings are L2-normalised, so squared L2 distance ↔ cosine similarity:
    #   cos_sim = 1 - (dist² / 2)
    similarity_score: float = distance_to_similarity(nearest_dist)

    # ── Step 4: Apply threshold ───────────────────────────────────────────────
    matched: bool = similarity_score > SEMANTIC_MATCH_THRESHOLD

    if matched:
        logger.info(
            "Semantic match detected. Score=%.4f  Threshold=%.2f  IndexPos=%d",
            similarity_score,
            SEMANTIC_MATCH_THRESHOLD,
            int(indices[0][0]),
        )
    else:
        logger.debug(
            "No semantic match. Score=%.4f  Threshold=%.2f",
            similarity_score,
            SEMANTIC_MATCH_THRESHOLD,
        )

    return SemanticCheckResult(
        similarity_score=round(similarity_score, 6),
        matched=matched,
        nearest_distance=round(nearest_dist, 6),
    )
