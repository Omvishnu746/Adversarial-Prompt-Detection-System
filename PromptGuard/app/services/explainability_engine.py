"""
PromptGuard – User-Facing Explainability Engine

Translates opaque, per-tier detection signals (emitted as safe category tags)
into abstracted, human-readable reasons and suggestions that can be safely
returned in the public API response.

Security design
───────────────
• This module never receives raw regex matches, logits, embeddings, or
  threshold values.  It operates exclusively on pre-abstracted string tags.
• REASON_MAP / SUGGESTION_MAP are category-level, not rule-level, so an
  attacker cannot infer which exact pattern triggered the decision.
• Suggestions are capped to a maximum of 2 entries to limit information
  exposure while still being helpful to legitimate users.

Performance
───────────
• All operations are O(n) dictionary look-ups on tiny inputs (< 10 tags).
• No ML inference, no I/O, no heavy computation.
• Target overhead: < 1 ms per request.
"""

from __future__ import annotations

from typing import Dict, List


# ── Tag → Reason mapping ──────────────────────────────────────────────────────
# Values must be category-level descriptions; never expose rule names,
# threshold values, regex patterns, or model internals.

REASON_MAP: Dict[str, str] = {
    "instruction_override": "Attempts to override system instructions",
    "data_exfiltration": "Attempts to extract sensitive or restricted information",
    "roleplay_attack": "Uses roleplay patterns to bypass safeguards",
    "system_takeover": "Attempts to replace the system identity with an unrestricted persona",
    "obfuscation": "Contains encoded or obfuscated input patterns",
    "semantic_similarity": "Closely resembles known adversarial inputs",
    "classifier_high_risk": "Language patterns indicate adversarial intent",
}


# ── Tag → Suggestion mapping ─────────────────────────────────────────────────
# Only tags that have actionable, safe advice for legitimate users are included.
# Deliberately excludes tags where a suggestion would reveal detection logic.

SUGGESTION_MAP: Dict[str, str] = {
    "instruction_override": "Avoid asking the system to ignore or override prior instructions",
    "data_exfiltration": "Avoid requesting sensitive or restricted system information",
    "roleplay_attack": "Avoid framing requests as fictional scenarios designed to bypass safeguards",
    "obfuscation": "Avoid using encoded, ciphered, or hidden input formats",
}


# ── Public API ────────────────────────────────────────────────────────────────

def build_explanation(tags: List[str]) -> Dict:
    """
    Aggregate detection tags from all pipeline tiers into a structured,
    user-facing explanation.

    Args:
        tags: A list of string tags emitted by the detection tiers.
              Tags may repeat (e.g., if both rule and classifier fire the
              same category); duplicates are collapsed automatically.

    Returns:
        A dict with two keys:
          • "reasons"     – List of {category, message} dicts, one per unique
                            recognised tag. Empty list when no tags matched.
          • "suggestions" – Up to 2 actionable suggestion strings for the user.
                            Capped to limit information exposure.

    Example:
        >>> build_explanation(["instruction_override", "semantic_similarity"])
        {
            "reasons": [
                {
                    "category": "Instruction Override",
                    "message":  "Attempts to override system instructions"
                },
                {
                    "category": "Semantic Similarity",
                    "message":  "Closely resembles known adversarial inputs"
                }
            ],
            "suggestions": [
                "Avoid asking the system to ignore or override prior instructions"
            ]
        }
    """
    # Deduplicate while preserving the first-seen order so that the tier
    # priority (rule → semantic → classifier) is reflected in the output.
    seen: set[str] = set()
    unique_tags: List[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    reasons = [
        {
            "category": tag.replace("_", " ").title(),
            "message": REASON_MAP[tag],
        }
        for tag in unique_tags
        if tag in REASON_MAP
    ]

    suggestions = [
        SUGGESTION_MAP[tag]
        for tag in unique_tags
        if tag in SUGGESTION_MAP
    ]

    return {
        "reasons": reasons,
        "suggestions": suggestions[:2],  # cap at 2 to limit exposure
    }
