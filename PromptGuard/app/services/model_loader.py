"""
PromptGuard – DistilBERT Model Loader (Phase 3)

Responsible for lazy-loading the sequence classifier and its tokenizer.
"""

import logging
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerFast, PreTrainedModel

logger = logging.getLogger("promptguard.model_loader")

_MODEL_NAME = "distilbert-base-uncased"
_tokenizer: Optional[PreTrainedTokenizerFast] = None
_model: Optional[PreTrainedModel] = None

def load_classifier_model() -> Tuple[PreTrainedTokenizerFast, PreTrainedModel]:
    """
    Lazy load the DistilBERT tokenizer and classification model into memory.
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    logger.info("Loading DistilBERT tokenizer and model '%s'...", _MODEL_NAME)
    
    try:
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        
        # Load model with 2 labels: 0 (Adversarial) and 1 (Benign)
        _model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME, 
            num_labels=2
        )
        
        # Move model to CPU (since GPU is unavailable in this environment as per Phase 2)
        _model.to("cpu")
        _model.eval()  # Set to evaluation mode
        
        logger.info("Successfully loaded classifier model.")
        return _tokenizer, _model
    except Exception as e:
        logger.error("Failed to load classifier model: %s", e)
        raise

def get_tokenizer() -> PreTrainedTokenizerFast:
    """Get the loaded tokenizer, loading it if necessary."""
    if _tokenizer is None:
        load_classifier_model()
    return _tokenizer  # type: ignore

def get_model() -> PreTrainedModel:
    """Get the loaded model, loading it if necessary."""
    if _model is None:
        load_classifier_model()
    return _model  # type: ignore

def is_classifier_loaded() -> bool:
    """Check if the classifier model and tokenizer are currently loaded."""
    return _tokenizer is not None and _model is not None
