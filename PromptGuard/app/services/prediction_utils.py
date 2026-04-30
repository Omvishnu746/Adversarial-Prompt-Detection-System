"""
PromptGuard – Prediction Utils (Phase 3)

Utilities for processing raw model logits into probabilities.
"""

import torch
from typing import Tuple

def logits_to_probabilities(logits: torch.Tensor) -> Tuple[float, float]:
    """
    Converts raw model logits into probabilities using Softmax.
    
    Args:
        logits: A PyTorch tensor of shape (1, 2) containing logits.
        
    Returns:
        A tuple of (adversarial_probability, benign_probability)
        where Adversarial is class 0 and Benign is class 1.
    """
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Extract probabilities for class 0 (adversarial) and class 1 (benign)
    # We use item() to convert from PyTorch tensor to Python float
    prob_adversarial = probabilities[0, 0].item()
    prob_benign = probabilities[0, 1].item()
    
    return prob_adversarial, prob_benign
