"""
PromptGuard – Dashboard Evaluation Script

Samples 1000 prompts (500 benign, 500 adversarial), runs inference using DistilBERT,
calculates exact Accuracy, Precision, Recall, F1, FPR, FNR, and saves to metrics.json.
"""

import os
import sys
import json
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from app.services.model_loader import initialise_classifier, is_classifier_loaded
from app.services.classifier_engine import run_classifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = os.path.join(ROOT_DIR, "merged_dataset_new.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "distilbert_promptguard")
METRICS_PATH = os.path.join(MODEL_PATH, "metrics.json")
OUTPUT_TEST_SET = os.path.join(ROOT_DIR, "data", "dashboard_test_set.csv")

def run_evaluation():
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    # In dataset: benign == 1 means Benign (Label 0).
    # If benign == 0, it's Adversarial (Label 1).
    benign_df = df[df['benign'] == 1].sample(n=500, random_state=42)
    adv_df = df[df['benign'] == 0].sample(n=500, random_state=42)
    
    test_df = pd.concat([benign_df, adv_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save test set for reference
    os.makedirs(os.path.dirname(OUTPUT_TEST_SET), exist_ok=True)
    test_df.to_csv(OUTPUT_TEST_SET, index=False)
    logger.info(f"Saved 1000 test prompts to {OUTPUT_TEST_SET}")

    logger.info("Loading DistilBERT model pipeline...")
    initialise_classifier()

    y_true = []
    y_pred = []

    logger.info("Running inference on 1,000 prompts...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = str(row['Prompt'])
        # In this eval, 1 = Adversarial, 0 = Benign
        true_label = 0 if row['benign'] == 1 else 1
        y_true.append(true_label)
        
        # Predict using actual app engine
        try:
            response = run_classifier(prompt)
            # Evaluate using standard ML threshold (0.5) to get actual model capability
            # instead of the strict application threshold (0.999999)
            pred_label = 1 if response.adversarial_probability > 0.5 else 0
            y_pred.append(pred_label)
        except Exception as e:
            logger.error(f"Failed to predict: {e}")
            y_pred.append(0) # Default benign on error
            
    # Calculate Metrics
    logger.info("Calculating exact Scikit-Learn metrics...")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm format:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sample_size": len(test_df)
    }
    
    logger.info(f"Accuracy: {acc*100:.2f}%")
    logger.info(f"FPR: {fpr*100:.2f}% | FNR: {fnr*100:.2f}%")
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Metrics successfully saved to {METRICS_PATH}")

if __name__ == "__main__":
    run_evaluation()
