"""
PromptGuard – Text Preprocessing Service

Pipeline:
  1. Unicode normalisation (NFKC)
  2. Lowercase conversion
  3. Extra whitespace removal
  4. Base64 chunk detection + decoding
  5. Leetspeak normalisation

All functions are pure and stateless for easy unit-testing.
"""

import re
import base64
import unicodedata
from typing import Optional


# ── Leetspeak map ────────────────────────────────────────────────────────────
_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "i",
    "(": "c",
    "+": "t",
}

# Regex: detect base64-looking tokens (≥16 chars, only base64 alphabet)
_BASE64_PATTERN = re.compile(r"\b([A-Za-z0-9+/]{16,}={0,2})\b")


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Apply NFKC Unicode normalisation to collapse compatibility characters."""
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
      - NFKC unicode normalisation
      - Lowercase
      - Collapse extra whitespace (spaces, tabs, newlines → single space)
      - Strip leading / trailing whitespace
    """
    text = normalize_unicode(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def detect_base64(text: str) -> Optional[str]:
    """
    Scan text for base64-encoded chunks.
    Returns the decoded plaintext of the *first* valid match, or None.

    A match is only returned if the decoded bytes are valid UTF-8.
    """
    for match in _BASE64_PATTERN.finditer(text):
        candidate = match.group(1)
        # Pad if necessary
        padding = 4 - len(candidate) % 4
        if padding != 4:
            candidate += "=" * padding
        try:
            decoded = base64.b64decode(candidate).decode("utf-8")
            return decoded
        except Exception:
            continue
    return None


def normalize_leetspeak(text: str) -> str:
    """
    Replace common leetspeak substitutions with their plain-text equivalents.
    Operates character-by-character using the static _LEET_MAP.
    """
    return "".join(_LEET_MAP.get(ch, ch) for ch in text)


def preprocess(text: str) -> str:
    """
    Composed preprocessing entry-point used by the inference pipeline.

    Order:
      1. clean_text  (unicode → lower → whitespace)
      2. detect_base64 and inline-replace if found
      3. normalize_leetspeak
    """
    text = clean_text(text)

    # If any base64 chunk is found, replace the entire text with its decoded form
    decoded = detect_base64(text)
    if decoded:
        text = clean_text(decoded)

    text = normalize_leetspeak(text)
    return text
