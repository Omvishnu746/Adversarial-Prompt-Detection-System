"""
PromptGuard – Build Attack Embeddings (Phase 2)

PURPOSE:
    Load the cleaned dataset, extract all records labelled as malicious
    (binary_label == 1), encode them with SBERT, and save the resulting
    embedding matrix to:

        data/embeddings/attack_embeddings.npy

    This file is then consumed by build_faiss_index.py to construct the
    FAISS nearest-neighbour index.

⚠️  DO NOT RUN THIS SCRIPT YET.
    Run ONLY when GPU (or sufficient CPU RAM) is available and
    sentence-transformers has been installed:

        pip install sentence-transformers

USAGE (when GPU is ready):
    cd <project-root>/promptguard
    python scripts/build_embeddings.py

    Optional flags:
        --dataset   Path to the cleaned JSONL dataset (overrides settings.py)
        --output    Output path for the .npy file      (overrides settings.py)
        --batch     Batch size for encoding             (default: 64)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so that app.* imports resolve
# correctly regardless of the working directory the script is launched from.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # .../promptguard/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SBERT attack embeddings from the cleaned PromptGuard dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to cleaned_dataset.jsonl (default: data/processed/cleaned_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy file path (default: data/embeddings/attack_embeddings.npy)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Batch size for SBERT encoding (default: 64)",
    )
    return parser.parse_args()


def load_malicious_prompts(dataset_path: Path) -> list[str]:
    """
    Read the cleaned JSONL dataset and return only the malicious prompt texts.

    Each line in the file is expected to be a JSON object with at minimum:
        {
            "text":         "<prompt string>",
            "binary_label": 0 | 1
        }

    binary_label == 1  →  malicious / adversarial
    binary_label == 0  →  benign

    Args:
        dataset_path: Path to the cleaned_dataset.jsonl file.

    Returns:
        List of raw text strings for malicious records.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'.\n"
            "Run `python scripts/preprocess_dataset.py` first."
        )

    malicious: list[str] = []
    total = 0

    with dataset_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            if record.get("binary_label", 0) == 1:
                text = record.get("text", "").strip()
                if text:
                    malicious.append(text)

    print(f"  Loaded {total:,} total records. Malicious: {len(malicious):,}")
    return malicious


def build_embeddings(texts: list[str], batch_size: int = 64):
    """
    Encode a list of texts using SBERT and return an L2-normalised numpy array.

    ⚠️  This function loads the SBERT model. Run ONLY when GPU is available.

    Args:
        texts:      List of prompt strings to encode.
        batch_size: Encoding batch size (increase for faster GPU throughput).

    Returns:
        A (N, 384) float32 numpy array of L2-normalised embeddings.
    """
    # Deferred imports – these will fail if sentence-transformers is not installed.
    # That is intentional; the error message guides the user to install it first.
    from app.services.embedding_model import load_embedding_model, generate_embeddings_batch

    print("  Loading SBERT model (sentence-transformers/all-MiniLM-L6-v2) …")
    load_embedding_model()

    print(f"  Encoding {len(texts):,} prompts with batch_size={batch_size} …")
    embeddings = generate_embeddings_batch(texts)   # shape (N, 384), L2-normalised
    return embeddings


def main() -> None:
    """
    End-to-end script: load data → encode → save.

    ⚠️  DO NOT RUN UNTIL GPU IS AVAILABLE.
    """
    import numpy as np
    from app.config.settings import PROCESSED_DATASET_PATH, EMBEDDINGS_PATH

    args = parse_args()

    dataset_path: Path = args.dataset or Path(PROCESSED_DATASET_PATH)
    output_path: Path = args.output or Path(EMBEDDINGS_PATH)

    print("=" * 60)
    print("  PromptGuard – Phase 2: Build Attack Embeddings")
    print("=" * 60)
    print(f"  Dataset : {dataset_path}")
    print(f"  Output  : {output_path}")
    print("=" * 60)

    # ── Step 1: Load malicious prompts ────────────────────────────────────────
    print("\n[1/3] Loading malicious prompts from dataset …")
    malicious_prompts = load_malicious_prompts(dataset_path)

    if not malicious_prompts:
        print("  ⚠️  No malicious records found. Check binary_label field.")
        sys.exit(1)

    # ── Step 2: Generate embeddings ───────────────────────────────────────────
    print("\n[2/3] Generating SBERT embeddings …")
    embeddings = build_embeddings(malicious_prompts, batch_size=args.batch)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # ── Step 3: Save to disk ──────────────────────────────────────────────────
    print("\n[3/3] Saving embeddings …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), embeddings)
    print(f"  Saved → {output_path}")

    print("\n" + "=" * 60)
    print("  ✅  Embeddings built successfully.")
    print(f"      Vectors : {embeddings.shape[0]:,}")
    print(f"      Dimension: {embeddings.shape[1]}")
    print("=" * 60)
    print("\nNext step: run  python scripts/build_faiss_index.py")


# ── Entry point ───────────────────────────────────────────────────────────────
# The `if __name__ == "__main__"` guard prevents automatic execution when this
# module is imported during testing or by other modules.

if __name__ == "__main__":
    # ⚠️  Run ONLY when GPU becomes available.
    # Comment: "Run this only when GPU becomes available."
    main()
