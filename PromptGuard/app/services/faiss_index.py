"""
PromptGuard – FAISS Index Manager (Phase 2)

Handles loading, saving, and querying a FAISS flat-L2 index that stores
pre-computed embeddings of known adversarial prompts.

IMPORTANT:
  • The index is NOT loaded at import time.
  • Call load_faiss_index() explicitly (or it will be called on demand).
  • Do NOT call build_faiss_index() here – that belongs to the build script.
  • Embeddings inside the index MUST be L2-normalised before insertion so
    that L2 distance directly maps to cosine similarity.

DO NOT execute this file directly during Phase 2 scaffolding.
Run only when GPU and pre-built embeddings are available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("promptguard.faiss_index")

# ── Paths ─────────────────────────────────────────────────────────────────────
# These paths are resolved relative to the promptguard package root at import
# time, but the files themselves are not opened until load_faiss_index() is
# called explicitly.

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]          # .../promptguard/
FAISS_INDEX_PATH: Path = _PACKAGE_ROOT / "data" / "embeddings" / "faiss_index.bin"
ATTACK_EMBEDDINGS_PATH: Path = _PACKAGE_ROOT / "data" / "embeddings" / "attack_embeddings.npy"

# ── Module-level singleton ────────────────────────────────────────────────────
_index: Optional[object] = None   # type: ignore[type-arg]  # faiss.Index once loaded


# ── Public API ────────────────────────────────────────────────────────────────

def load_faiss_index(index_path: Optional[Path] = None) -> None:
    """
    Load a pre-built FAISS index from disk into the module-level singleton.

    The index file is expected at ``data/embeddings/faiss_index.bin`` by
    default, though a custom path can be supplied.

    This function is idempotent – calling it multiple times is safe.

    Prerequisites:
        pip install faiss-cpu   # CPU-only
        # OR
        pip install faiss-gpu   # once GPU is available

    ⚠️  Build the index first by running:
        python scripts/build_faiss_index.py

    Args:
        index_path: Optional override for the default FAISS index file path.

    Raises:
        FileNotFoundError: If the index file does not exist on disk.
        RuntimeError:      If faiss is not installed.
    """
    global _index

    if _index is not None:
        logger.debug("FAISS index already loaded – skipping reload.")
        return

    path = Path(index_path) if index_path else FAISS_INDEX_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'. "
            "Generate it first by running:\n"
            "    python scripts/build_faiss_index.py"
        )

    try:
        import faiss  # noqa: PLC0415

        logger.info("Loading FAISS index from: %s", path)
        _index = faiss.read_index(str(path))
        logger.info(
            "FAISS index loaded. Total vectors: %d  Dimension: %d",
            _index.ntotal,
            _index.d,
        )
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. "
            "Run `pip install faiss-cpu` before activating semantic detection."
        ) from exc


def save_faiss_index(index: object, output_path: Optional[Path] = None) -> Path:
    """
    Persist a FAISS index object to disk.

    Used by scripts/build_faiss_index.py – NOT called automatically.

    Args:
        index:       A faiss.Index object to serialise.
        output_path: Destination file path. Defaults to ``FAISS_INDEX_PATH``.

    Returns:
        The resolved Path where the index was written.
    """
    import faiss  # noqa: PLC0415

    path = Path(output_path) if output_path else FAISS_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(path))  # type: ignore[arg-type]
    logger.info("FAISS index saved at %s  (%d vectors)", path, index.ntotal)  # type: ignore[attr-defined]
    return path


def is_index_loaded() -> bool:
    """Return True if a FAISS index is resident in memory."""
    return _index is not None


def get_index_size() -> int:
    """
    Return the number of vectors stored in the loaded index.

    Returns:
        Integer count, or 0 if the index has not been loaded yet.
    """
    if _index is None:
        return 0
    return _index.ntotal  # type: ignore[attr-defined]


def search_index(
    query_vector: np.ndarray,
    top_k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for the ``top_k`` nearest neighbours to
    ``query_vector``.

    The index is loaded on demand if not already resident in memory, but it
    is recommended to call ``load_faiss_index()`` at application startup.

    Args:
        query_vector: 1-D float32 numpy array of shape (D,) representing the
                      L2-normalised query embedding.
        top_k:        Number of nearest neighbours to retrieve.

    Returns:
        A tuple ``(distances, indices)`` where:
          - distances: shape (1, top_k) float32 array of squared L2 distances.
          - indices:   shape (1, top_k) int64 array of vector indices in the
                       index.
        If the index is empty or the result is invalid, distances are ``inf``
        and indices are ``-1``.

    Raises:
        RuntimeError: If faiss is not installed.
        FileNotFoundError: If the index file cannot be found on lazy load.
    """
    if _index is None:
        logger.warning(
            "search_index() called before load_faiss_index(). Loading on demand."
        )
        load_faiss_index()

    if _index.ntotal == 0:  # type: ignore[union-attr]
        logger.warning("FAISS index is empty – returning default (no match) result.")
        inf_dist = np.full((1, top_k), np.inf, dtype=np.float32)
        neg_idx = np.full((1, top_k), -1, dtype=np.int64)
        return inf_dist, neg_idx

    # FAISS expects a 2-D array: shape (n_queries, D)
    query_2d = query_vector.reshape(1, -1).astype(np.float32)

    distances, indices = _index.search(query_2d, top_k)  # type: ignore[union-attr]
    return distances, indices


def build_flat_l2_index(embeddings: np.ndarray) -> object:
    """
    Create a new FAISS IndexFlatL2 from a matrix of L2-normalised embeddings.

    This helper is called by ``scripts/build_faiss_index.py`` – it is NOT
    called automatically by this module.

    Args:
        embeddings: 2-D float32 numpy array of shape (N, D).

    Returns:
        A populated faiss.IndexFlatL2 object ready for saving or querying.

    Raises:
        RuntimeError: If faiss is not installed.
        ValueError:   If embeddings array is empty or has wrong shape.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"embeddings must be a non-empty 2-D array; got shape {embeddings.shape}."
        )

    try:
        import faiss  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Run `pip install faiss-cpu`."
        ) from exc

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    logger.info(
        "Built IndexFlatL2 with %d vectors of dimension %d.", index.ntotal, dimension
    )
    return index
