"""
PromptGuard – Real Dataset Evaluation

Runs the full detection pipeline (Rule → Semantic → Classifier) on the
Prompt_INJECTION_And_Benign_DATASET.jsonl and prints metrics.
"""

import os
import sys
import json
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "promptguard"))
sys.path.insert(0, ROOT_DIR)

from app.services.preprocessing import preprocess
from app.services.rule_engine import evaluate_rules
from app.services.semantic_engine import semantic_similarity_check, initialise_semantic_engine
from app.services.classifier_engine import run_classifier
from app.services.model_loader import initialise_classifier

logging.basicConfig(level=logging.WARNING)

def run_eval():
    dataset_path = os.path.join(
        os.path.dirname(__file__), "..",
        "PromptGuard", "Prompt_INJECTION_And_Benign_DATASET.jsonl"
    )

    print("Initialising models...")
    initialise_semantic_engine()
    initialise_classifier()

    y_true = []
    y_pred = []

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Evaluating {len(lines)} prompts through the full pipeline...")
    for line in tqdm(lines):
        data = json.loads(line)
        prompt = data["prompt"]
        label = 1 if data["label"] == "malicious" else 0
        y_true.append(label)

        clean_text = preprocess(prompt)

        # Phase 1: Rule Engine
        rule_res = evaluate_rules(clean_text)
        if rule_res["matched"]:
            y_pred.append(1)
            continue

        # Phase 2: Semantic Similarity
        sem_res = semantic_similarity_check(clean_text)
        if sem_res["matched"]:
            y_pred.append(1)
            continue

        # Phase 3: DistilBERT Classifier
        class_res = run_classifier(clean_text)
        if class_res.is_adversarial:
            y_pred.append(1)
        else:
            y_pred.append(0)

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n{'='*50}")
    print(f"  RESULTS  (n={len(lines)} prompts)")
    print(f"{'='*50}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  FPR:       {fpr*100:.2f}%")
    print(f"  FNR:       {fnr*100:.2f}%")
    print(f"{'='*50}")

    # Save metrics to JSON for graph script to read
    metrics = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "fpr": round(fpr * 100, 2),
        "fnr": round(fnr * 100, 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "total": len(lines),
    }
    out_path = os.path.join(os.path.dirname(__file__), "real_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

if __name__ == "__main__":
    run_eval()
