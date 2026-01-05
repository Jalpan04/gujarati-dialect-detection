import re
import unicodedata

def normalize_gujarati_text(text):
    """
    Normalizes Gujarati text for dialect classification.
    
    1. Unicode Normalization (NFC)
    2. Zero Width Joiner/Non-Joiner removal
    3. Whitespace cleanup
    4. Optional: Filtering non-target characters (keeping Gujarati, English, digits, punctuation)
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Unicode Normalization
    text = unicodedata.normalize('NFC', text)
    
    # 2. Remove ZWJ (u200d) and ZWNJ (u200c) common in Indic typing
    text = text.replace('\u200d', '').replace('\u200c', '')
    
    # 3. Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_noise(text):
    """
    Aggressive cleaning for training.
    Removes things that obscure dialectal features.
    """
    # Remove HTML tags (if any slipped through)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove brackets/citations [1], [kha]
    text = re.sub(r'\[.*?\]', '', text)
    
    return text

if __name__ == "__main__":
    # Test
    sample = "મુંબઈ (Mumbai) [૧] એક     બીજું શું? \u200d"
    print(f"Original: '{sample}'")
    print(f"Normalized: '{normalize_gujarati_text(sample)}'")
