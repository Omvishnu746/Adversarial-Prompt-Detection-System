"""
PromptGuard – DistilBERT Training Script (Phase 3)

Prepares the dataset, tokenizes with chunking/truncation, and sets up the 
HuggingFace Trainer for the DistilBERT sequence classifier.

Note: As per requirements, this script sets up the logic but does NOT 
execute `trainer.train()` yet.
"""

import sys
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# Add project root to path if needed
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.settings import PROCESSED_DATASET_PATH, SBERT_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_distilbert")

# DistilBERT model
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = _PROJECT_ROOT / "models" / "distilbert_weights"


def load_dataset() -> Tuple[list[str], list[int]]:
    """Loads text and labels from the JSONL dataset."""
    texts = []
    labels = []
    
    logger.info("Loading dataset from %s", PROCESSED_DATASET_PATH)
    
    with open(PROCESSED_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            
            # Extract text
            text = record.get("prompt_cleaned", record.get("text", "")).strip()
            if not text:
                continue
                
            # Extract label: 1 = Benign, 0 = Adversarial
            attack_labels = record.get("attack_labels", {})
            label = attack_labels.get("is_benign", 1)
            
            texts.append(text)
            labels.append(label)
            
    logger.info("Loaded %d records. (Benign: %d, Adversarial: %d)", 
                len(texts), sum(labels), len(labels) - sum(labels))
                
    return texts, labels


def prepare_datasets(texts: list[str], labels: list[int]) -> DatasetDict:
    """Create Stratified 80/10/10 Train/Val/Test split."""
    logger.info("Creating stratified splits (80/10/10)...")
    
    # First split: 80% Train, 20% Temp
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.20, stratify=labels, random_state=42
    )
    
    # Second split: 50% Val, 50% Test (from the 20% Temp -> 10% Val, 10% Test)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    # Create HuggingFace Datasets
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    logger.info("Split sizes -> Train: %d, Val: %d, Test: %d", 
                len(train_dataset), len(val_dataset), len(test_dataset))
                
    return dataset_dict


def main():
    print("="*60)
    print(" PromptGuard Phase 3 - DistilBERT Training Setup")
    print("="*60)
    
    # 1. Load Data
    texts, labels = load_dataset()
    
    # 2. Split Data
    dataset = prepare_datasets(texts, labels)
    
    # 3. Load Tokenizer & Tokenize
    logger.info("Loading tokenizer '%s'...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        # Truncate to max length. In a real dynamic sliding-window training setup,
        # we would map multiple chunks to the same label, but standard fine-tuning
        # often just truncates to 512 tokens. For Phase 3, we truncate for training,
        # but our inference engine uses the sliding window!
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 4. Load Model
    logger.info("Loading model '%s'...", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 6. Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Trainer successfully configured.")
    logger.info("NOTE: training is purposefully NOT executed in this step.")
    
    # --- UNCOMMENT TO TRAIN ---
    # logger.info("Starting training...")
    # trainer.train()
    # 
    # logger.info("Saving best model...")
    # trainer.save_model(str(OUTPUT_DIR / "final"))
    # tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

if __name__ == "__main__":
    from typing import Tuple
    main()
