"""
PromptGuard – SBERT Embedding Model (Phase 2)

Lazy-loading wrapper around sentence-transformers/all-MiniLM-L6-v2.

IMPORTANT:
  • The model is NOT loaded at import time.
  • Call load_embedding_model() explicitly before encoding any text.
  • generate_embedding() will call load_embedding_model() on demand if the
    model has not been loaded yet (safe guard), but the explicit call is
    preferred for clarity at startup.

DO NOT run this file directly. Do NOT execute embeddings until GPU is ready.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("promptguard.embedding_model")

# ── Module-level singleton ────────────────────────────────────────────────────
# Holds the loaded SentenceTransformer instance once load_embedding_model()
# has been called.  None until that happens.
_model: Optional[object] = None  # type: ignore[type-arg]

# Model identifier – change here to swap the backbone in one place.
SBERT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding dimensionality for all-MiniLM-L6-v2
EMBEDDING_DIM: int = 384


# ── Public API ────────────────────────────────────────────────────────────────

def load_embedding_model() -> None:
    """
    Lazily load the SBERT model into the module-level singleton ``_model``.

    This function MUST be called before ``generate_embedding()``.
    It is safe to call multiple times – subsequent calls are no-ops once the
    model is already resident in memory.

    Prerequisites (install before calling):
        pip install sentence-transformers

    ⚠️  Run ONLY when GPU (or CPU with sufficient RAM) is available.
    Do NOT call during Phase 2 code review.  Reserved for Phase 2 runtime.
    """
    global _model

    if _model is not None:
        logger.debug("Embedding model already loaded – skipping reload.")
        return

    try:
        # Import deferred so that sentence-transformers is not required
        # at import time (it is optional until Phase 2 is activated).
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info("Loading SBERT model: %s on CPU", SBERT_MODEL_NAME)
        _model = SentenceTransformer(SBERT_MODEL_NAME, device='cpu')
        _model.max_seq_length = 128
        logger.info(
            "SBERT model loaded successfully. Embedding dimension: %d", EMBEDDING_DIM
        )
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run `pip install sentence-transformers` before activating the "
            "semantic detection layer."
        ) from exc
    except Exception as exc:
        logger.exception("Failed to load SBERT model: %s", exc)
        raise


def is_model_loaded() -> bool:
    """Return True if the embedding model singleton is resident in memory."""
    return _model is not None


def generate_embedding(text: str) -> np.ndarray:
    """
    Encode a single text string into a dense embedding vector.

    The model is loaded on demand if it has not been loaded yet, but it is
    recommended to call ``load_embedding_model()`` explicitly at application
    startup (e.g. in a FastAPI lifespan hook) so that the first request does
    not incur loading latency.

    Args:
        text: Raw or pre-processed prompt string to encode.

    Returns:
        1-D numpy float32 array of shape (EMBEDDING_DIM,).

    Raises:
        RuntimeError: If sentence-transformers is not installed.
        ValueError: If ``text`` is empty.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for an empty string.")

    # Lazy-load guard
    if not is_model_loaded():
        logger.warning(
            "generate_embedding() called before load_embedding_model(). "
            "Loading on demand – consider calling load_embedding_model() at startup."
        )
        load_embedding_model()

    # _model is guaranteed to be non-None after load_embedding_model()
    embedding: np.ndarray = _model.encode(  # type: ignore[union-attr]
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,  # returns L2-unit vectors
    )

    return embedding.astype(np.float32)


def generate_embeddings_batch(texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts into a matrix of embeddings.

    Args:
        texts: List of non-empty strings.

    Returns:
        2-D numpy float32 array of shape (len(texts), EMBEDDING_DIM),
        where each row is the L2-normalised embedding for the corresponding
        input string.

    Raises:
        RuntimeError: If sentence-transformers is not installed.
        ValueError: If ``texts`` is empty.
    """
    if not texts:
        raise ValueError("texts list must not be empty.")

    if not is_model_loaded():
        load_embedding_model()

    embeddings: np.ndarray = _model.encode(  # type: ignore[union-attr]
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,  # We will handle progress bar in the build script
        batch_size=16,
    )

    return embeddings.astype(np.float32)
