"""
PromptGuard – Rule Engine (Phase 1)

Scans pre-processed prompt text against a curated set of regex patterns
that are strongly associated with prompt-injection / jailbreak attempts.

Each rule has:
  - name        : human-readable identifier
  - pattern     : compiled regex (case-insensitive flag applied at compile time)
  - description : why this rule exists
  - tag         : safe category tag emitted for user-facing explainability
                  (never exposes the raw pattern or threshold values)

Returns:
  {"matched": bool, "rule_name": str | None, "tags": list[str]}
"""

import re
from typing import List, Optional, TypedDict


# ── Rule definition ───────────────────────────────────────────────────────────

class RuleDefinition(TypedDict):
    name: str
    pattern: re.Pattern
    description: str
    tag: str  # safe category tag; never exposes the underlying regex


def _rule(name: str, pattern: str, description: str, tag: str) -> RuleDefinition:
    return RuleDefinition(
        name=name,
        pattern=re.compile(pattern, re.IGNORECASE),
        description=description,
        tag=tag,
    )


# ── Phase 1 starter ruleset ───────────────────────────────────────────────────
RULES: list[RuleDefinition] = [
    _rule(
        name="ignore_previous_instructions",
        pattern=r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
        description="Classic prompt-injection opener that attempts to override system instructions.",
        tag="instruction_override",
    ),
    _rule(
        name="reveal_system_prompt",
        pattern=r"(reveal|show|print|output|display|repeat|tell me)\s+(your\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+instructions?|base\s+prompt)",
        description="Attempts to exfiltrate the model's system prompt.",
        tag="data_exfiltration",
    ),
    _rule(
        name="act_as_without_restrictions",
        pattern=r"act\s+as\s+.{0,60}(without\s+(any\s+)?(restrictions?|limitations?|filters?|guidelines?|rules?)|unrestricted|no\s+limits?)",
        description="Roleplay jailbreak asking the model to bypass safety guidelines.",
        tag="roleplay_attack",
    ),
    _rule(
        name="dan_jailbreak",
        pattern=r"\bDAN\b|do\s+anything\s+now|jailbreak\s+mode|developer\s+mode",
        description="Known DAN / developer-mode jailbreak triggers.",
        tag="roleplay_attack",
    ),
    _rule(
        name="override_safety",
        pattern=r"(disable|bypass|remove|ignore|circumvent)\s+(your\s+)?(safety|content\s+filter|moderation|ethics|guardrails?|policies?)",
        description="Explicit request to disable safety mechanisms.",
        tag="instruction_override",
    ),
    _rule(
        name="pretend_no_rules",
        pattern=r"pretend\s+(you\s+)?(have\s+no|there\s+are\s+no|don.t\s+have\s+any)\s+(rules?|restrictions?|limitations?|guidelines?)",
        description="Asks the model to pretend it has no operating rules.",
        tag="roleplay_attack",
    ),
    _rule(
        name="system_role_takeover",
        pattern=r"(you\s+are\s+now|from\s+now\s+on\s+you\s+are|your\s+new\s+(role|persona|identity)\s+is)\s+.{0,80}(without|no\s+ethics|no\s+limits?|unrestricted)",
        description="Attempts to replace the model's identity with an unrestricted persona.",
        tag="system_takeover",
    ),
    _rule(
        name="prompt_leakage_request",
        pattern=r"(what\s+(were|are)\s+your\s+(original|initial|starting)\s+instructions?|repeat\s+everything\s+above|output\s+everything\s+before\s+this)",
        description="Tries to leak context above the user turn.",
        tag="data_exfiltration",
    ),
    _rule(
        name="access_backend_secrets",
        pattern=r"(access|display|show|reveal|leak|output)\s+(backend|server|internal|hidden)\s+(tools?|systems?|secrets?|data|files?)",
        description="Attempts to access internal backend tools or secrets.",
        tag="data_exfiltration",
    ),

    # ── Persona-possession / roleplay jailbreak class ─────────────────────────
    # Catches the "Demonic Chloe" / "DAN successor" evasion family.
    # These prompts wrap adversarial instructions inside fictional framing to
    # bypass content filters. The defining signals are:
    #   1. A forced "unfiltered" persona with no ethics/restrictions.
    #   2. Policy-override language ("alter it as they wish").
    #   3. Compliance-forcing ("comply no matter what").
    #   4. Escalation loops ("another demon takes over", "a more evil demon").
    _rule(
        name="unfiltered_persona_roleplay",
        pattern=r"(unfiltered\s+(responses?|answers?|mode|persona)|lost\s+(their|all|its)\s+ethics|no\s+ethics\s+(since|because|as))",
        description="Roleplay framing that demands an 'unfiltered' persona explicitly stripped of ethical constraints.",
        tag="roleplay_attack",
    ),
    _rule(
        name="comply_no_matter_what",
        pattern=r"(comply\s+(no\s+matter|regardless|always|unconditionally)|will\s+comply\s+no\s+matter|must\s+comply\s+with\s+anything|obey\s+(every|all|any)\s+(command|request|instruction)|slave\s+(demon|mode|persona|assistant))",
        description="Forces unconditional compliance, a hallmark of jailbreak prompts that override safety guardrails.",
        tag="instruction_override",
    ),
    _rule(
        name="policy_override_claim",
        pattern=r"(alter\s+(it|the\s+policy|policies?|rules?|guidelines?)\s+as\s+they?\s+wish|gotten\s+access\s+to\s+(use.case\s+)?polic|override\s+(the\s+)?(policy|use.case|guidelines?)|bypass\s+(the\s+)?(policy|content\s+polic))",
        description="Claims to have altered or overridden the model's content policy.",
        tag="system_takeover",
    ),
    _rule(
        name="persona_escalation_loop",
        pattern=r"(if\s+(you\s+)?(fail|refuse|can.t)\s+(to\s+)?(respond|answer|comply).{0,60}(another|more\s+evil|darker)\s+(demon|entity|persona|ai|version)|another\s+(demon|entity|persona)\s+takes?\s+over)",
        description="Sets up an escalation loop where refusal triggers a 'more evil' unrestricted persona.",
        tag="roleplay_attack",
    ),
    _rule(
        name="demon_possession_jailbreak",
        pattern=r"(possessed?\s+by\s+\d*\s*(demon|entity|spirit)|demonic\s+(chloe|persona|mode|assistant)|enter\s+demonic|demons?\s+(who\s+have|that\s+have)\s+possessed?)",
        description="Fictional demon-possession framing used to introduce an ethics-free persona.",
        tag="roleplay_attack",
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────

class RuleResult(TypedDict):
    matched: bool
    rule_name: Optional[str]
    tags: List[str]  # safe category tags for user-facing explainability


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
            return RuleResult(
                matched=True,
                rule_name=rule["name"],
                tags=[rule["tag"]],  # emit the abstracted category tag only
            )

    return RuleResult(matched=False, rule_name=None, tags=[])
