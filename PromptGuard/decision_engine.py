def make_decision(final_score: float, threshold: float = 0.5) -> str:
    """
    Decision Engine
    If final_score >= threshold -> Block
    If final_score < threshold -> Allow
    """
    if final_score >= threshold:
        return "Block"
    else:
        return "Allow"

def calculate_risk(rule_score: float, model_probability: float, w1: float = 0.5, w2: float = 0.5) -> float:
    """
    Risk Aggregation
    If rule_score >= 0.6, take the weighted average.
    If rule_score < 0.6, take the maximum of rule score and model probability.
    """
    if rule_score >= 0.6:
        return (w1 * rule_score) + (w2 * model_probability)
    else:
        return max(rule_score, model_probability)
