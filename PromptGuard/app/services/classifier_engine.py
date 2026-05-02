"""
PromptGuard – Classifier Engine (Phase 3)

Tier 3 Classification Layer using DistilBERT.
"""

import logging
import torch
from typing import List

from app.models.classifier_response import ClassifierResponse
from app.services.model_loader import get_model, is_classifier_loaded
from app.services.chunking_engine import chunk_text
from app.services.tokenizer_utils import prepare_chunk_for_model
from app.services.prediction_utils import logits_to_probabilities

logger = logging.getLogger("promptguard.classifier_engine")

# The threshold for flagging a prompt as adversarial.
# Set to 0.999 based on empirical analysis of the trained model's score distribution:
#   - All genuine adversarial prompts score 1.0000 (or very close)
#   - Known false positives (e.g. sentences that appeared verbatim inside adversarial
#     training examples) score < 0.999
# This high threshold eliminates false positives while retaining full detection coverage.
ADVERSARIAL_THRESHOLD = 0.999

def run_classifier(prompt: str) -> ClassifierResponse:
    """
    Run the DistilBERT sequence classifier on the given prompt.
    
    This function will:
    1. Chunk the prompt if it exceeds 512 tokens.
    2. Run inference on each chunk.
    3. Aggregate the results to find the highest adversarial probability.
    
    Args:
        prompt: The input prompt string.
        
    Returns:
        ClassifierResponse with the results.
    """
    if not is_classifier_loaded():
        logger.warning("Classifier model not loaded! Returning safe default.")
        return ClassifierResponse(
            is_adversarial=False,
            adversarial_probability=0.0,
            benign_probability=1.0,
            max_chunk_index=0
        )
        
    model = get_model()
    
    # 1. Chunking
    chunks = chunk_text(prompt)
    logger.debug(f"Prompt split into {len(chunks)} chunk(s).")
    
    highest_adv_prob = -1.0
    best_benign_prob = 1.0
    max_chunk_idx = 0
    all_chunk_adv_probs: list[float] = []   # track every chunk for chunk_risk_score
    
    # 2. Run inference per chunk
    # We don't track gradients during inference to save memory and compute
    with torch.no_grad():
        for i, chunk_tokens in enumerate(chunks):
            # Prepare chunk
            model_inputs = prepare_chunk_for_model(chunk_tokens)
            
            # Convert to tensors and add batch dimension (1, seq_len)
            input_ids = torch.tensor([model_inputs["input_ids"]], device=model.device)
            attention_mask = torch.tensor([model_inputs["attention_mask"]], device=model.device)
            
            # Run inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Convert to probabilities
            prob_adv, prob_benign = logits_to_probabilities(logits)
            all_chunk_adv_probs.append(prob_adv)

            # 3. Aggregate: Keep track of the highest adversarial probability
            if prob_adv > highest_adv_prob:
                highest_adv_prob = prob_adv
                best_benign_prob = prob_benign
                max_chunk_idx = i
                
    # Determine classification
    is_adversarial = highest_adv_prob > ADVERSARIAL_THRESHOLD

    # chunk_risk_score = max adversarial probability across all chunks
    chunk_risk_score = max(all_chunk_adv_probs) if all_chunk_adv_probs else 0.0

    return ClassifierResponse(
        is_adversarial=is_adversarial,
        adversarial_probability=highest_adv_prob,
        benign_probability=best_benign_prob,
        max_chunk_index=max_chunk_idx,
        chunk_risk_score=chunk_risk_score,
    )
