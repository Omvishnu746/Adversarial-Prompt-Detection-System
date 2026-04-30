"""
PromptGuard – DistilBERT Evaluation Script (Phase 3)

Evaluates the fine-tuned DistilBERT model against the Test dataset.
Calculates Accuracy, Precision, Recall, and F1-Score.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path if needed
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# We can reuse the load_dataset and prepare_datasets functions from train_distilbert
# In a real environment, they might be refactored into a shared data_utils module.
from scripts.train_distilbert import load_dataset, prepare_datasets, MODEL_NAME, OUTPUT_DIR

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_model")

def compute_metrics(eval_pred):
    """
    Computes Accuracy, Precision, Recall, and F1-Score.
    Because 1=Benign and 0=Adversarial, we usually care about the performance 
    on predicting Adversarial (class 0). However, macro or binary with pos_label=0
    can be used. We'll return binary metrics focusing on Adversarial detection.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # We set pos_label=0 because identifying Adversarial (0) is our main objective
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=0
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    print("="*60)
    print(" PromptGuard Phase 3 - DistilBERT Evaluation")
    print("="*60)
    
    # 1. Load Data
    texts, labels = load_dataset()
    dataset = prepare_datasets(texts, labels)
    test_dataset = dataset["test"]
    
    # 2. Check if model exists
    final_model_path = OUTPUT_DIR / "final"
    if not final_model_path.exists():
        logger.warning("Fine-tuned model not found at %s.", final_model_path)
        logger.warning("Evaluation will run on the BASE pre-trained model for demonstration.")
        model_path_to_load = MODEL_NAME
    else:
        logger.info("Loading fine-tuned model from %s", final_model_path)
        model_path_to_load = str(final_model_path)
        
    # 3. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_path_to_load)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_to_load, num_labels=2)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    logger.info("Tokenizing test dataset...")
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # 4. Initialize Trainer for Evaluation
    trainer = Trainer(
        model=model,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    
    # 5. Evaluate
    logger.info("Starting evaluation...")
    results = trainer.evaluate()
    
    print("\n" + "="*40)
    print(" EVALUATION RESULTS (Adversarial = 0)")
    print("="*40)
    print(f" Accuracy  : {results.get('eval_accuracy', 0.0):.4f}")
    print(f" Precision : {results.get('eval_precision', 0.0):.4f}")
    print(f" Recall    : {results.get('eval_recall', 0.0):.4f}")
    print(f" F1-Score  : {results.get('eval_f1', 0.0):.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
