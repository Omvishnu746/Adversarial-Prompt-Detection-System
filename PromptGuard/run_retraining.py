import sys
from train_model import train

if __name__ == '__main__':
    print("Starting retraining with new dataset: merged_dataset_new.csv")
    try:
        train('merged_dataset_new.csv')
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
