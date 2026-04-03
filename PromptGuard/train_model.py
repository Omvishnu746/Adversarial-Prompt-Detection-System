import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from data_loader import load_and_preprocess_data

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train(csv_path: str, output_dir: str = './model_output'):
    """
    Fine-tune DistilBERT for binary classification on the adversarial metric.
    Input -> Prompt
    Output -> adversarial probability
    """
    # 1. Load data
    df = load_and_preprocess_data(csv_path)
    
    # 2. Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        df['Prompt'].tolist(), 
        df['adversarial'].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    
    # 3. Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 4. Tokenize texts
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=512)
    
    # 5. Create datasets
    train_dataset = PromptDataset(train_encodings, y_train)
    val_dataset = PromptDataset(val_encodings, y_val)
    
    # 6. Initialize model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # 7. Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch", # Optional: can be disabled if val set is extremely large
        save_strategy="epoch"
    )
    
    # 8. Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    
    # 9. Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Example usage: train("dataset.csv")
    pass
