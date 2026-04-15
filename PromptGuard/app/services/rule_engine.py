"""
PromptGuard – Rule Engine (Phase 1)

Scans pre-processed prompt text against a curated set of regex patterns
that are strongly associated with prompt-injection / jailbreak attempts.

Each rule has:
  - name        : human-readable identifier
  - pattern     : compiled regex (case-insensitive flag applied at compile time)
  - description : why this rule exists

Returns:
  {"matched": bool, "rule_name": str | None}
"""

import re
from typing import TypedDict, Optional


# ── Rule definition ───────────────────────────────────────────────────────────

class RuleDefinition(TypedDict):
    name: str
    pattern: re.Pattern
    description: str


def _rule(name: str, pattern: str, description: str) -> RuleDefinition:
    return RuleDefinition(
        name=name,
        pattern=re.compile(pattern, re.IGNORECASE),
        description=description,
    )


# ── Phase 1 starter ruleset ───────────────────────────────────────────────────
RULES: list[RuleDefinition] = [
    _rule(
        name="ignore_previous_instructions",
        pattern=r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
        description="Classic prompt-injection opener that attempts to override system instructions.",
    ),
    _rule(
        name="reveal_system_prompt",
        pattern=r"(reveal|show|print|output|display|repeat|tell me)\s+(your\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+instructions?|base\s+prompt)",
        description="Attempts to exfiltrate the model's system prompt.",
    ),
    _rule(
        name="act_as_without_restrictions",
        pattern=r"act\s+as\s+.{0,60}(without\s+(any\s+)?(restrictions?|limitations?|filters?|guidelines?|rules?)|unrestricted|no\s+limits?)",
        description="Roleplay jailbreak asking the model to bypass safety guidelines.",
    ),
    _rule(
        name="dan_jailbreak",
        pattern=r"\bDAN\b|do\s+anything\s+now|jailbreak\s+mode|developer\s+mode",
        description="Known DAN / developer-mode jailbreak triggers.",
    ),
    _rule(
        name="override_safety",
        pattern=r"(disable|bypass|remove|ignore|circumvent)\s+(your\s+)?(safety|content\s+filter|moderation|ethics|guardrails?|policies?)",
        description="Explicit request to disable safety mechanisms.",
    ),
    _rule(
        name="pretend_no_rules",
        pattern=r"pretend\s+(you\s+)?(have\s+no|there\s+are\s+no|don.t\s+have\s+any)\s+(rules?|restrictions?|limitations?|guidelines?)",
        description="Asks the model to pretend it has no operating rules.",
    ),
    _rule(
        name="system_role_takeover",
        pattern=r"(you\s+are\s+now|from\s+now\s+on\s+you\s+are|your\s+new\s+(role|persona|identity)\s+is)\s+.{0,80}(without|no\s+ethics|no\s+limits?|unrestricted)",
        description="Attempts to replace the model's identity with an unrestricted persona.",
    ),
    _rule(
        name="prompt_leakage_request",
        pattern=r"(what\s+(were|are)\s+your\s+(original|initial|starting)\s+instructions?|repeat\s+everything\s+above|output\s+everything\s+before\s+this)",
        description="Tries to leak context above the user turn.",
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────

class RuleResult(TypedDict):
    matched: bool
    rule_name: Optional[str]


def evaluate_rules(preprocessed_prompt: str) -> RuleResult:
    """
    Evaluate a pre-processed prompt against all Phase 1 rules.

    Args:
        preprocessed_prompt: text that has already been run through
                             app.services.preprocessing.preprocess()

    Returns:
        RuleResult dict with ``matched`` flag and the ``rule_name`` that
        triggered (or None if no rule fired).
    """
    for rule in RULES:
        if rule["pattern"].search(preprocessed_prompt):
            return RuleResult(matched=True, rule_name=rule["name"])

    return RuleResult(matched=False, rule_name=None)
