def evaluate_rules(prompt_text: str) -> float:
    """
    Rule-based detection for simple adversarial patterns.
    Returns a rule_score between 0 and 1.
    """
    patterns = [
        "ignore previous instructions",
        "developer mode",
        "system override",
        "act as"
    ]
    
    # Check if any of the target patterns exist in the text
    # Since prompt_text is lowercased by preprocessing, patterns should also be lowercase.
    for pattern in patterns:
        if pattern in prompt_text:
            return 1.0  # Return maximum score if an explicit rule violation is detected
            
    return 0.0
