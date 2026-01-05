
import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Config
MODEL_PATH = "models/muril_finetuned/final"
DATA_FILE = "D:/gujarati_project_data/dataset/raw_text/wikisource_dialect_corpus_v2_1767513116.csv"
SAMPLE_SIZE = 1000 # Test on 1000 samples for speed

def eval_muril():
    print(f"--- Evaluating MuRIL Model ---")
    
    # 1. Load Data
    print("Loading Data...")
    df = pd.read_csv(DATA_FILE)
    df = df[df['dialect_label'].isin(['standard', 'kathiawadi', 'saurashtra'])]
    df['label'] = df['dialect_label'].map({
        'standard': 0, 
        'kathiawadi': 1,
        'saurashtra': 1
    })
    
    # Simulate the Train/Test split we used during training
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Take a sample for quick evaluation
    test_df = test_df.sample(min(SAMPLE_SIZE, len(test_df)), random_state=42)
    print(f"Evaluating on {len(test_df)} held-out samples...")

    # 2. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    
    predictions = []
    ground_truth = test_df['label'].tolist()
    texts = test_df['text_raw'].tolist()
    
    # 3. Inference Loop
    print("Running Inference...")
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            
    # 4. Metrics
    acc = accuracy_score(ground_truth, predictions)
    print("\n" + "="*30)
    print(f"✅ Accuracy: {acc:.2%}")
    print("="*30)
    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions, target_names=['Standard', 'Kathiawadi']))

if __name__ == "__main__":
    eval_muril()
