import pandas as pd

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV dataset, validate expected columns, and ignore unused columns for now.
    Creates a derived training label 'adversarial'.
    """
    df = pd.read_csv(csv_path)
    
    expected_columns = [
        "Prompt", "jailbreak", "role_manipulation", "prompt_injection",
        "indirect_injection", "obfuscation", "benign"
    ]
    
    # Validate expected columns exist (case-sensitive)
    for col in expected_columns:
        if col not in df.columns:
            # Try case-insensitive fallback if exact match wasn't found
            col_lower = col.lower()
            df.columns = [c.lower() if c.lower() == col_lower else c for c in df.columns]
            if col_lower not in df.columns:
                raise ValueError(f"Missing expected column: {col}")
            # Ensure the DataFrame has the expected key now
            df.rename(columns={col_lower: col}, inplace=True)
            
    # Create derived training label
    # adversarial = 1 if (jailbreak == 1 OR prompt_injection == 1)
    # adversarial = 0 if benign == 1
    def get_adversarial_label(row):
        if row['jailbreak'] == 1 or row['prompt_injection'] == 1:
            return 1
        elif row['benign'] == 1:
            return 0
        return 0  # Default fallback if row matches nothing
        
    df['adversarial'] = df.apply(get_adversarial_label, axis=1)
    
    # Ignore unused columns for now (role_manipulation, indirect_injection, obfuscation)
    # Return only the necessary columns for training: Prompt and the derived adversarial label
    return df[['Prompt', 'adversarial']]
