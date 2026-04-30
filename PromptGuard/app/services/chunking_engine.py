"""
PromptGuard – Sliding Window Chunking Engine (Phase 3)

Splits long token sequences into overlapping chunks to bypass the 512-token limit.
"""

from typing import List
from app.services.tokenizer_utils import tokenize_text

def chunk_text(text: str, window_size: int = 512, stride: int = 256) -> List[List[int]]:
    """
    Splits a prompt into overlapping token chunks.
    
    Args:
        text: The input prompt string.
        window_size: Maximum token length per chunk (default 512).
                     Note: The actual window for raw tokens is window_size - 2 
                     to account for [CLS] and [SEP].
        stride: The number of tokens to shift the window by for overlaps (default 256).
        
    Returns:
        A list of token ID lists (chunks).
    """
    # We need room for [CLS] and [SEP]
    max_raw_tokens = window_size - 2
    
    # Tokenize the text without special tokens
    tokens = tokenize_text(text)
    
    chunks = []
    
    # If text is shorter than window_size, return original tokens as one chunk
    if len(tokens) <= max_raw_tokens:
        chunks.append(tokens)
        return chunks
        
    # Generate overlapping chunks
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = start_idx + max_raw_tokens
        chunk = tokens[start_idx:end_idx]
        chunks.append(chunk)
        
        # Move the window forward by stride
        start_idx += stride
        
    return chunks
