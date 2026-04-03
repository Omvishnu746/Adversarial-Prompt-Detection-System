import re

def preprocess_prompt(prompt_text: str) -> str:
    """
    Convert text to lowercase, remove extra whitespace, and normalize prompt text.
    """
    # Convert text to lowercase
    prompt_text = prompt_text.lower()
    
    # Remove extra whitespace (replace multiple spaces/newlines with single space)
    prompt_text = re.sub(r'\s+', ' ', prompt_text)
    
    # Normalize prompt text (strip leading and trailing spaces)
    prompt_text = prompt_text.strip()
    
    return prompt_text
