"""
PromptGuard – Build FAISS Index (Phase 2)

PURPOSE:
    Load the pre-computed attack_embeddings.npy matrix (produced by
    build_embeddings.py), create a FAISS IndexFlatL2 index, and save it to:

        data/embeddings/faiss_index.bin

    The FAISS index is the core nearest-neighbour lookup structure used by
    the semantic_engine.py at request time.

⚠️  DO NOT RUN THIS SCRIPT YET.
    Build the embeddings first (run build_embeddings.py),
    then run this script ONLY when faiss-cpu is installed:

        pip install faiss-cpu

USAGE (when embeddings are ready):
    cd <project-root>/promptguard
    python scripts/build_faiss_index.py

    Optional flags:
        --embeddings  Path to attack_embeddings.npy (overrides settings.py)
        --output      Output path for faiss_index.bin (overrides settings.py)

EXECUTION ORDER:
    Step 1  →  python scripts/preprocess_dataset.py
    Step 2  →  python scripts/build_embeddings.py          ← requires sentence-transformers
    Step 3  →  python scripts/build_faiss_index.py         ← requires faiss-cpu  (THIS FILE)
    Step 4  →  Start FastAPI server; semantic engine is now active.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # .../promptguard/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS IndexFlatL2 from pre-computed attack embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to attack_embeddings.npy (default: data/embeddings/attack_embeddings.npy)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for faiss_index.bin (default: data/embeddings/faiss_index.bin)",
    )
    return parser.parse_args()


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """
    Load the pre-built attack embeddings from a .npy file.

    Args:
        embeddings_path: Path to the attack_embeddings.npy file.

    Returns:
        2-D float32 numpy array of shape (N, D).

    Raises:
        FileNotFoundError: If the .npy file does not exist.
        ValueError:        If the array is not 2-D.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at '{embeddings_path}'.\n"
            "Run `python scripts/build_embeddings.py` first."
        )

    embeddings = np.load(str(embeddings_path)).astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2-D embeddings array, got shape {embeddings.shape}."
        )

    print(f"  Loaded embeddings: shape={embeddings.shape}  dtype={embeddings.dtype}")
    return embeddings


def main() -> None:
    """
    End-to-end script: load embeddings → build FAISS index → save.

    ⚠️  DO NOT RUN UNTIL faiss-cpu IS INSTALLED AND EMBEDDINGS ARE BUILT.
    """
    from app.services.faiss_index import build_flat_l2_index, save_faiss_index
    from app.config.settings import EMBEDDINGS_PATH, FAISS_INDEX_PATH

    args = parse_args()

    embeddings_path: Path = args.embeddings or Path(EMBEDDINGS_PATH)
    output_path: Path = args.output or Path(FAISS_INDEX_PATH)

    print("=" * 60)
    print("  PromptGuard – Phase 2: Build FAISS Index")
    print("=" * 60)
    print(f"  Embeddings : {embeddings_path}")
    print(f"  Output     : {output_path}")
    print("=" * 60)

    # ── Step 1: Load attack embeddings ────────────────────────────────────────
    print("\n[1/3] Loading attack embeddings …")
    embeddings = load_embeddings(embeddings_path)

    # ── Step 2: Build FAISS IndexFlatL2 ──────────────────────────────────────
    print(f"\n[2/3] Building FAISS IndexFlatL2 (dim={embeddings.shape[1]}) …")
    index = build_flat_l2_index(embeddings)
    print(f"  Total vectors in index: {index.ntotal:,}")  # type: ignore[attr-defined]

    # ── Step 3: Save index ────────────────────────────────────────────────────
    print("\n[3/3] Saving FAISS index …")
    saved_path = save_faiss_index(index, output_path=output_path)
    print(f"  Saved at {saved_path}")

    metadata_path = output_path.parent / "index_metadata.json"
    import json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "embedding_dimension": int(embeddings.shape[1]),
            "vector_count": int(index.ntotal)
        }, f, indent=4)
    print(f"  Saved metadata → {metadata_path}")

    print("\n" + "=" * 60)
    print("  ✅  FAISS index built successfully.")
    print(f"      Vectors  : {index.ntotal:,}")          # type: ignore[attr-defined]
    print(f"      Dimension: {embeddings.shape[1]}")
    print("=" * 60)
    print(
        "\nSemantic engine is now ready.\n"
        "Start the API server with:  python run.py\n"
        "The /check_prompt endpoint will automatically activate the semantic layer."
    )


# ── Entry point ───────────────────────────────────────────────────────────────
# The `if __name__ == "__main__"` guard prevents automatic execution when this
# module is imported during testing or by other modules.

if __name__ == "__main__":
    # ⚠️  Run ONLY after build_embeddings.py has completed successfully.
    # Comment: "Run this only when GPU becomes available and embeddings are built."
    main()
