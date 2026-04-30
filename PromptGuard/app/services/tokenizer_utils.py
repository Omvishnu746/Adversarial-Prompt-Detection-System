"""
PromptGuard – Tokenizer Utils (Phase 3)

Helper functions for interacting with the DistilBERT tokenizer.
"""

from typing import List, Dict, Any
from app.services.model_loader import get_tokenizer

def tokenize_text(text: str) -> List[int]:
    """
    Tokenize a string into a list of token IDs without adding special tokens yet,
    because we will chunk them first and then add special tokens to each chunk.
    """
    tokenizer = get_tokenizer()
    # add_special_tokens=False so we can chunk raw tokens and add [CLS]/[SEP] later
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens

def prepare_chunk_for_model(token_ids: List[int], max_length: int = 512) -> Dict[str, Any]:
    """
    Prepare a raw token ID chunk for the DistilBERT model.
    Adds [CLS] at the beginning, [SEP] at the end, and creates attention masks.
    """
    tokenizer = get_tokenizer()
    
    # Add special tokens: [CLS] token_ids [SEP]
    cls_token = tokenizer.cls_token_id
    sep_token = tokenizer.sep_token_id
    
    # Ensure it doesn't exceed max_length (512)
    # Subtract 2 to leave room for [CLS] and [SEP]
    max_tokens = max_length - 2
    truncated_ids = token_ids[:max_tokens]
    
    final_input_ids = [cls_token] + truncated_ids + [sep_token]
    attention_mask = [1] * len(final_input_ids)
    
    # We could pad here, but we can also just return the list 
    # since PyTorch can handle dynamic batching or we process 1 by 1.
    # Let's pad it to max_length for consistency if doing batch processing.
    padding_length = max_length - len(final_input_ids)
    
    if padding_length > 0:
        pad_token = tokenizer.pad_token_id
        final_input_ids.extend([pad_token] * padding_length)
        attention_mask.extend([0] * padding_length)
        
    return {
        "input_ids": final_input_ids,
        "attention_mask": attention_mask
    }
