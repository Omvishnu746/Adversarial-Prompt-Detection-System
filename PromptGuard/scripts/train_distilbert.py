"""
PromptGuard – DistilBERT Training Script (CPU Optimized)

Prepares the dataset, tokenizes with chunking/truncation, and sets up the 
HuggingFace Trainer for the DistilBERT sequence classifier.
"""

import os
import sys
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# Add project root to path if needed
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.settings import PROCESSED_DATASET_PATH

# Configure logging
log_dir = _PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "distilbert_training.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train_distilbert_cpu")

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = _PROJECT_ROOT / "models" / "distilbert_promptguard"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

def load_dataset():
    """Loads text and labels from the JSONL dataset."""
    texts = []
    labels = []
    
    logger.info(f"Loading dataset from {PROCESSED_DATASET_PATH}")
    
    with open(PROCESSED_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            
            # Using prompt_cleaned as the pre-processed version of prompt
            text = record.get("prompt_cleaned", record.get("prompt_original", "")).strip()
            if not text:
                continue
                
            attack_labels = record.get("attack_labels", {})
            label = attack_labels.get("is_benign", 1)
            
            texts.append(text)
            labels.append(label)
            
    total_size = len(texts)
    benign_samples = sum(labels)
    adv_samples = total_size - benign_samples
    
    print("="*40)
    print(f"Total dataset size: {total_size}")
    print(f"Number of benign samples: {benign_samples}")
    print(f"Number of adversarial samples: {adv_samples}")
    print("="*40)
                
    return texts, labels


def prepare_datasets(texts, labels):
    """Create Stratified 80/10/10 Train/Val/Test split."""
    logger.info("Creating stratified splits (Train 80% / Val 10% / Test 10%)...")
    
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.20, stratify=labels, random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    return dataset_dict


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }


def main():
    print("="*60)
    print(" PromptGuard Phase 3 - DistilBERT Training (CPU)")
    print("="*60)
    
    # 1. Load Data
    texts, labels = load_dataset()
    
    # 2. Split Data
    dataset = prepare_datasets(texts, labels)
    
    # 3. Load Tokenizer & Tokenize
    logger.info(f"Loading tokenizer '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        # Tokenize with sliding window chunking to handle long sequences
        tokenized = tokenizer(
            examples["text"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
            stride=32
        )
        # Map original labels to new chunks
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["label"] = [examples["label"][i] for i in sample_mapping]
        return tokenized
        
    logger.info("Tokenizing datasets with sliding window chunking...")
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text", "label"]
    )
    
    # 4. Load Model
    logger.info(f"Loading model '{MODEL_NAME}'...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        warmup_steps=500,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        use_cpu=True, # Force CPU execution
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 6. Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 7. Train Model
    logger.info("Starting training...")
    resume_from_checkpoint = False
    if OUTPUT_DIR.exists():
        checkpoints = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
        if checkpoints:
            resume_from_checkpoint = True
            logger.info("Found existing checkpoints. Will resume from the latest checkpoint.")
            
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info("Saving best model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # 8. Evaluate Model on Test Set
    logger.info("Evaluating model on test set...")
    test_results = trainer.predict(tokenized_datasets["test"])
    metrics = test_results.metrics
    
    # Extract the custom metrics
    final_metrics = {
        "accuracy": metrics.get("test_accuracy", 0.0),
        "precision": metrics.get("test_precision", 0.0),
        "recall": metrics.get("test_recall", 0.0),
        "f1_score": metrics.get("test_f1_score", 0.0)
    }
    
    print("\n--- TEST METRICS ---")
    for k, v in final_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    logger.info(f"Saving metrics to {METRICS_PATH}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)
        
    print("Training and evaluation complete. System ready for Tier 4 Router implementation.")

if __name__ == "__main__":
    main()
