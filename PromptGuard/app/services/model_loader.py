"""
PromptGuard – DistilBERT Model Loader (Phase 3)

Loads the locally fine-tuned DistilBERT sequence classifier from disk.
Falls back to the base HuggingFace model if the local weights are not found.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)

from app.config.settings import CLASSIFIER_MODEL_PATH

logger = logging.getLogger("promptguard.model_loader")

_BASE_MODEL_NAME = "distilbert-base-uncased"   # fallback only
_tokenizer: Optional[PreTrainedTokenizerFast] = None
_model: Optional[PreTrainedModel] = None


def load_classifier_model() -> Tuple[PreTrainedTokenizerFast, PreTrainedModel]:
    """
    Load the fine-tuned DistilBERT tokenizer and sequence classifier into memory.

    Prefers the locally trained weights at ``CLASSIFIER_MODEL_PATH``.
    Falls back to the base ``distilbert-base-uncased`` weights from HuggingFace
    if the local directory does not exist (useful for development / CI).
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    local_path = Path(CLASSIFIER_MODEL_PATH)
    if local_path.exists() and any(local_path.iterdir()):
        model_source = str(local_path)
        logger.info("Loading fine-tuned DistilBERT from local path: %s", model_source)
    else:
        model_source = _BASE_MODEL_NAME
        logger.warning(
            "Local model not found at '%s'. Falling back to base model '%s'.",
            CLASSIFIER_MODEL_PATH,
            _BASE_MODEL_NAME,
        )

    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_source)
        _model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=2,
        )
        _model.to("cpu")
        _model.eval()
        logger.info("Classifier model loaded successfully (source: %s).", model_source)
        return _tokenizer, _model
    except Exception as exc:
        logger.error("Failed to load classifier model: %s", exc)
        raise


def initialise_classifier() -> None:
    """
    Eagerly load the classifier at application startup.
    Call this from the FastAPI lifespan handler so the first request
    does not pay the cold-start penalty.
    """
    load_classifier_model()


def get_tokenizer() -> PreTrainedTokenizerFast:
    """Return the loaded tokenizer, loading it on first call."""
    if _tokenizer is None:
        load_classifier_model()
    return _tokenizer  # type: ignore[return-value]


def get_model() -> PreTrainedModel:
    """Return the loaded model, loading it on first call."""
    if _model is None:
        load_classifier_model()
    return _model  # type: ignore[return-value]


def is_classifier_loaded() -> bool:
    """Return True if both model and tokenizer are resident in memory."""
    return _tokenizer is not None and _model is not None
