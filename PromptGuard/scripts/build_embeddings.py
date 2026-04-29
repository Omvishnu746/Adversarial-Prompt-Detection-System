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
            
            # Use attack_labels['is_benign'] == 0 as the malicious indicator
            attack_labels = record.get("attack_labels", {})
            is_malicious = attack_labels.get("is_benign", 1) == 0
            # Some versions might have binary_label explicitly
            if record.get("binary_label") == 1 or is_malicious:
                text = record.get("prompt_cleaned", record.get("text", "")).strip()
                if text:
                    malicious.append(text)

    print(f"  Loaded {total:,} total records. Malicious: {len(malicious):,}")
    return malicious


def build_embeddings(texts: list[str], batch_size: int = 16, checkpoint_dir: Path = None):
    """
    Encode a list of texts using SBERT and return an L2-normalised numpy array.
    Supports checkpointing every 500 prompts and tqdm progress bar.
    """
    from app.services.embedding_model import load_embedding_model, generate_embeddings_batch
    import numpy as np
    from tqdm import tqdm
    import os

    print("  Loading SBERT model (sentence-transformers/all-MiniLM-L6-v2) on CPU …")
    load_embedding_model()

    print(f"  Encoding {len(texts):,} prompts with batch_size={batch_size} …")
    
    if checkpoint_dir is None:
        checkpoint_dir = Path("data/embeddings")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / "checkpoint_embeddings.npy"
    progress_file = checkpoint_dir / "checkpoint_progress.txt"
    
    start_idx = 0
    all_embeddings = []
    
    if checkpoint_file.exists() and progress_file.exists():
        with open(progress_file, "r") as f:
            start_idx = int(f.read().strip())
        all_embeddings = list(np.load(str(checkpoint_file)))
        print(f"  Resuming from checkpoint at index {start_idx}...")

    chunk_size = 500
    for i in tqdm(range(start_idx, len(texts), chunk_size), desc="Generating Embeddings", unit="chunk"):
        chunk_texts = texts[i : i + chunk_size]
        chunk_embeddings = generate_embeddings_batch(chunk_texts)
        all_embeddings.extend(chunk_embeddings)
        
        # Save checkpoint
        np.save(str(checkpoint_file), np.array(all_embeddings, dtype=np.float32))
        with open(progress_file, "w") as f:
            f.write(str(i + len(chunk_texts)))

    final_embeddings = np.array(all_embeddings, dtype=np.float32)
    # Cleanup checkpoints
    if checkpoint_file.exists():
        os.remove(checkpoint_file)
    if progress_file.exists():
        os.remove(progress_file)

    return final_embeddings


def main() -> None:
    """
    End-to-end script: load data → encode → save.
    """
    import numpy as np
    import time
    import logging
    from app.config.settings import PROCESSED_DATASET_PATH, EMBEDDINGS_PATH

    args = parse_args()

    dataset_path: Path = args.dataset or Path(PROCESSED_DATASET_PATH)
    output_path: Path = args.output or Path(EMBEDDINGS_PATH)
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "embedding_generation.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("  PromptGuard – Phase 2: Build Attack Embeddings")
    print("=" * 60)
    print(f"  Dataset : {dataset_path}")
    print(f"  Output  : {output_path}")
    print("=" * 60)

    logging.info("Embedding generation started.")
    start_time = time.time()

    # ── Step 1: Load malicious prompts ────────────────────────────────────────
    print("\n[1/3] Loading malicious prompts from dataset …")
    malicious_prompts = load_malicious_prompts(dataset_path)

    if not malicious_prompts:
        print("  WARNING: No malicious records found. Check binary_label field.")
        sys.exit(1)

    # ── Step 2: Generate embeddings ───────────────────────────────────────────
    print("\n[2/3] Generating SBERT embeddings …")
    checkpoint_dir = output_path.parent
    embeddings = build_embeddings(malicious_prompts, batch_size=16, checkpoint_dir=checkpoint_dir)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # ── Step 3: Save to disk ──────────────────────────────────────────────────
    print("\n[3/3] Saving embeddings …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), embeddings)
    print(f"  Saved → {output_path}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Completion time: {elapsed_time:.2f} seconds.")
    logging.info(f"Total processed prompts: {len(malicious_prompts)}.")
    logging.info("Embedding generation completed successfully.")

    print("\n" + "=" * 60)
    print("  SUCCESS: Embeddings built successfully.")
    print(f"      Vectors : {embeddings.shape[0]:,}")
    print(f"      Dimension: {embeddings.shape[1]}")
    print(f"      Elapsed Time: {elapsed_time:.2f} seconds")
    print("=" * 60)
    print("\nNext step: run  python scripts/build_faiss_index.py")


if __name__ == "__main__":
    main()
