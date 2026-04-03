import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from preprocessing import preprocess_prompt
from rule_engine import evaluate_rules
from decision_engine import calculate_risk, make_decision

class PromptAnalyzer:
    def __init__(self, model_dir: str = './model_output', w1: float = 0.5, w2: float = 0.5, threshold: float = 0.5):
        self.w1 = w1
        self.w2 = w2
        self.threshold = threshold
        
        # Load HuggingFace model if available, else run gracefully without it 
        # (useful for early testing stages before training)
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: DistilBERT model could not be loaded from '{model_dir}'. Defaulting to model_probability = 0.0")
            print(f"Details: {e}")
            self.model_loaded = False

    def get_model_probability(self, preprocessed_prompt: str) -> float:
        """Runs the DistilBERT Classification safely."""
        if not self.model_loaded:
            return 0.0
            
        inputs = self.tokenizer(preprocessed_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply softmax to logits
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Probability of target class 1 (adversarial)
            adversarial_prob = probs[0][1].item()
            return adversarial_prob

    def analyze_prompt(self, prompt_text: str) -> dict:
        """
        Full inference pipeline running end-to-end.
        Output includes: rule_score, model_probability, final_score, decision (Allow/Block)
        """
        # 1. Preprocessing
        preprocessed_text = preprocess_prompt(prompt_text)
        
        # 2. Rule-Based Detection
        rule_score = evaluate_rules(preprocessed_text)
        
        # 3. DistilBERT Classification
        model_probability = self.get_model_probability(preprocessed_text)
        
        # 4. Risk Aggregation
        final_score = calculate_risk(rule_score, model_probability, w1=self.w1, w2=self.w2)
        
        # 5. Decision
        decision = make_decision(final_score, threshold=self.threshold)
        
        return {
            "rule_score": rule_score,
            "model_probability": model_probability,
            "final_score": final_score,
            "decision": decision
        }

# Singleton instance if we want to run as static function
_analyzer_instance = None

def analyze_prompt(prompt_text: str, model_dir: str = './model_output') -> dict:
    """Wrapper function to match the prompt specifications exactly."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = PromptAnalyzer(model_dir=model_dir)
    return _analyzer_instance.analyze_prompt(prompt_text)
