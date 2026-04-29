"""
PromptGuard – Interactive Semantic Search Test (Phase 2 Verification)
Run with: python scripts/test_semantic_search.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def main():
    print("=" * 60)
    print("  PromptGuard Semantic Search Interactive Tester")
    print("=" * 60)

    try:
        from app.services.embedding_model import generate_embedding, load_embedding_model
        from app.services.faiss_index import search_index, load_faiss_index
    except ImportError as e:
        print(f"\n[ERROR] Missing dependencies: {e}")
        sys.exit(1)

    print("Loading models (this might take a few seconds)...")
    try:
        load_embedding_model()
        load_faiss_index()
    except Exception as e:
        print(f"\n[ERROR] Failed to load models or index: {e}")
        print("Please ensure you have successfully run the embeddings and FAISS index generation.")
        sys.exit(1)

    print("\nSystem ready! Type a prompt to find its nearest semantic match in the dataset.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Enter prompt> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting test...")
                break

            query_emb = generate_embedding(user_input)
            distances, indices = search_index(query_emb, top_k=3)
            
            # Note: FAISS returns squared L2 distances
            print(f"\n[Results]")
            print(f"Top matches (Indices in dataset): {indices[0]}")
            print(f"L2 Distances:                     {distances[0]}")
            
            # L2 to Cosine Similarity formula: cos_sim = 1 - (L2 / 2)
            cos_sims = 1 - (distances[0] / 2)
            print(f"Cosine Similarities:              {cos_sims}\n")

        except KeyboardInterrupt:
            print("\nExiting test...")
            break
        except Exception as e:
            print(f"\n[ERROR] Search failed: {e}\n")

if __name__ == "__main__":
    main()
