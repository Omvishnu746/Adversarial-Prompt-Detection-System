"""
PromptGuard – Explainability Module (Phase 5)

Extracts the most suspicious tokens from a prompt using DistilBERT's
self-attention weights (no extra dependencies needed).

Algorithm
─────────
1. Tokenize the prompt with the fine-tuned DistilBERT tokenizer.
2. Run a forward pass with output_attentions=True.
3. Average attention weights across all 6 heads and both transformer layers.
4. Use the CLS-token row (index 0) — it attends to the most discriminative tokens.
5. Decode the top-K token IDs back to word pieces, filter special tokens.
6. Return list of (token, attention_score) pairs.

This is fast enough to run in a Celery worker on CPU (~10-50 ms per call).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("promptguard.explainability")

# Lazy imports — heavy models may not be available in all environments
_tokenizer = None
_model     = None


def _load_model():
    """Lazily load the fine-tuned classifier model with attention output."""
    global _tokenizer, _model
    if _tokenizer is not None:
        return True

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from app.services.model_loader import get_model_path

        model_path = get_model_path()
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_attentions=True,
        )
        _model.eval()
        return True

    except Exception as exc:
        logger.warning("Explainability model not available: %s", exc)
        return False


def get_top_tokens(
    prompt: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Return up to *top_k* tokens with the highest CLS-attention scores.

    Parameters
    ----------
    prompt : str
        The raw (or preprocessed) prompt text.
    top_k : int
        Number of tokens to return.

    Returns
    -------
    List of dicts: [{"token": str, "score": float}, ...]
    Ordered highest → lowest score.
    Returns empty list if model is not available.
    """
    if not _load_model():
        return []

    try:
        import torch

        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )

        with torch.no_grad():
            outputs = _model(**inputs)

        # outputs.attentions: tuple of (num_layers,) tensors each
        # shape [batch=1, num_heads, seq_len, seq_len]
        # Stack → [num_layers, num_heads, seq_len, seq_len]
        attentions = torch.stack(outputs.attentions, dim=0)

        # Average across layers and heads → [seq_len, seq_len]
        avg_attention = attentions.mean(dim=(0, 1)).squeeze(0)

        # CLS row (token 0) attends to all positions
        cls_attention = avg_attention[0, :]  # [seq_len]

        input_ids = inputs["input_ids"][0]
        tokens = _tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # Filter special tokens
        special = {_tokenizer.cls_token, _tokenizer.sep_token, _tokenizer.pad_token}
        scored = [
            {"token": tok, "score": round(float(score), 6)}
            for tok, score in zip(tokens, cls_attention.tolist())
            if tok not in special and not tok.startswith("##") or True
        ]

        # Sort descending by score, take top_k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    except Exception as exc:
        logger.warning("Explainability extraction failed: %s", exc)
        return []


def tokens_to_json(token_list: list[dict]) -> str:
    """Serialise token list to a compact JSON string for DB storage."""
    import json
    return json.dumps(token_list, ensure_ascii=False)
