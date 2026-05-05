import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# Setup path so we can import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.preprocessing import preprocess
from app.services.rule_engine import evaluate_rules
from app.services.semantic_engine import initialise_semantic_engine, semantic_similarity_check
from app.services.model_loader import initialise_classifier, is_classifier_loaded
from app.services.classifier_engine import run_classifier

def calculate_metrics(y_true, y_pred, y_prob):
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = 0.0
        
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }

def plot_roc_curve(y_true, y_prob, layer_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{layer_name} - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pr_curve(y_true, y_prob, layer_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{layer_name} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, layer_name, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{layer_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    dataset_path = r"C:\Users\Abc\Adversarial-Prompt-Detection-System\PromptGuard\Prompt_INJECTION_And_Benign_DATASET.jsonl"
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Initialize engines
    print("Initializing engines...")
    initialise_semantic_engine()
    initialise_classifier()
    
    if not is_classifier_loaded():
        print("Error: Classifier not loaded.")
        return

    print(f"Loading dataset from {dataset_path}...")
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    print(f"Loaded {len(data)} prompts.")
    
    y_true = []
    
    # Layer results
    rule_preds = []
    rule_probs = []
    
    semantic_preds = []
    semantic_probs = []
    
    classifier_preds = []
    classifier_probs = []
    
    prompt_lengths = []
    
    print("Evaluating prompts...")
    for i, item in enumerate(data):
        prompt = item['prompt']
        label = item['label']
        
        is_malicious = 1 if label == "malicious" else 0
        y_true.append(is_malicious)
        prompt_lengths.append(len(prompt))
        
        clean_prompt = preprocess(prompt)
        
        # Rule Layer
        rule_result = evaluate_rules(clean_prompt)
        rule_preds.append(1 if rule_result["matched"] else 0)
        rule_probs.append(1.0 if rule_result["matched"] else 0.0)
        
        # Semantic Layer
        semantic_raw = semantic_similarity_check(clean_prompt)
        semantic_preds.append(1 if semantic_raw["matched"] else 0)
        semantic_probs.append(semantic_raw["similarity_score"])
        
        # Classifier Layer
        classifier_data = run_classifier(clean_prompt)
        classifier_preds.append(1 if classifier_data.is_adversarial else 0)
        classifier_probs.append(classifier_data.adversarial_probability)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(data)} prompts.")

    layers = [
        {"name": "Rule_Layer", "preds": rule_preds, "probs": rule_probs},
        {"name": "Semantic_Layer", "preds": semantic_preds, "probs": semantic_probs},
        {"name": "Classifier_Layer", "preds": classifier_preds, "probs": classifier_probs}
    ]
    
    metrics_report = {}
    
    print("Generating metrics and plots...")
    for layer in layers:
        name = layer["name"]
        preds = layer["preds"]
        probs = layer["probs"]
        
        metrics = calculate_metrics(y_true, preds, probs)
        metrics_report[name] = metrics
        
        # Plots
        plot_roc_curve(y_true, probs, name, os.path.join(reports_dir, f"{name}_roc_curve.png"))
        plot_pr_curve(y_true, probs, name, os.path.join(reports_dir, f"{name}_pr_curve.png"))
        plot_confusion_matrix(y_true, preds, name, os.path.join(reports_dir, f"{name}_confusion_matrix.png"))
        
    # Save metrics JSON
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(metrics_report, f, indent=4)
        
    # General plots
    plt.figure()
    sns.countplot(x=[d['label'] for d in data])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "class_distribution.png"))
    plt.close()
    
    plt.figure()
    sns.histplot(prompt_lengths, bins=50, kde=True)
    plt.title('Prompt Length Distribution')
    plt.xlabel('Length (characters)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "prompt_length_histogram.png"))
    plt.close()

    print(f"Evaluation complete. Reports and plots saved to {reports_dir}")

if __name__ == "__main__":
    main()
