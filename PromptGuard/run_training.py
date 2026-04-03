import sys
from train_model import train

if __name__ == '__main__':
    print("Starting training with dataset: merged_dataset.csv")
    try:
        train('merged_dataset.csv')
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
