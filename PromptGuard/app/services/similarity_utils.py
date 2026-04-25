"""
PromptGuard – Similarity Utility Functions (Phase 2)

Pure NumPy utilities used by the semantic engine to compute
cosine similarity and normalise embedding vectors.

All functions are stateless and have no side-effects.
NO models are loaded here.
"""

import numpy as np
from typing import Union


# ── Type alias ────────────────────────────────────────────────────────────────

Vector = Union[np.ndarray, list]


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_vector(vector: Vector) -> np.ndarray:
    """
    L2-normalise a single vector so that its Euclidean norm equals 1.0.

    Args:
        vector: A 1-D array-like of floats (raw embedding).

    Returns:
        A 1-D numpy float32 array of the same length, L2-normalised.
        If the vector has zero norm (all zeros), it is returned unchanged
        to avoid division by zero.
    """
    vec = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def normalize_vectors(matrix: Union[np.ndarray, list]) -> np.ndarray:
    """
    L2-normalise every row of a 2-D matrix of embeddings.

    Args:
        matrix: Shape (N, D) array-like where N = number of vectors,
                D = embedding dimension.

    Returns:
        A (N, D) float32 numpy array where each row has unit L2 norm.
    """
    mat = np.array(matrix, dtype=np.float32)
    # Compute per-row norms; keep dims so broadcasting works correctly
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Replace zero norms with 1.0 to avoid NaN on the division
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """
    Compute the cosine similarity between two vectors.

    Cosine similarity ∈ [-1, 1].
    A value of 1.0 means perfectly identical direction;
    0.0 means orthogonal; -1.0 means opposite.

    Args:
        vec_a: First embedding vector (1-D array-like).
        vec_b: Second embedding vector (1-D array-like, same length).

    Returns:
        Scalar float in the range [-1.0, 1.0].
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def distance_to_similarity(distance: float, scale: float = 1.0) -> float:
    """
    Convert an L2 distance returned by a FAISS index into a similarity score
    bounded in [0.0, 1.0].

    FAISS IndexFlatL2 returns squared L2 distances. When embeddings are
    L2-normalised, the relationship to cosine similarity is:

        cosine_similarity = 1 - (distance² / 2)

    This function applies that formula and clamps the result to [0.0, 1.0]
    to guard against floating-point edge cases.

    Args:
        distance: Squared L2 distance returned by FAISS (non-negative).
        scale:    Optional divisor applied before the formula, useful when
                  embeddings are NOT unit-normalised. Default is 1.0.

    Returns:
        Similarity score ∈ [0.0, 1.0].
        - 0.0 → completely dissimilar (distance = 2 on the unit sphere)
        - 1.0 → identical vectors   (distance = 0)
    """
    similarity = 1.0 - (distance / (2.0 * scale))
    # Clamp to valid range
    return float(np.clip(similarity, 0.0, 1.0))


def batch_cosine_similarity(query: Vector, matrix: np.ndarray) -> np.ndarray:
    """
    Compute the cosine similarity between a single query vector and every
    row in a matrix of stored embeddings.

    Args:
        query:  1-D array-like of shape (D,).
        matrix: 2-D numpy array of shape (N, D).

    Returns:
        1-D numpy float32 array of shape (N,) containing similarity scores.
    """
    q = normalize_vector(query)
    mat = normalize_vectors(matrix)
    return mat @ q   # shape (N,)
