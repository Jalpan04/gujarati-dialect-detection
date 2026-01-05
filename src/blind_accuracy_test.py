
import os
import sys
import random
import torch
import numpy as np
import librosa
import soundfile as sf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import GujaratiDialectPredictor
from data.augmentations import AudioAugmentor

def create_pseudo_test_set(source_dir, num_samples=50):
    """
    Creates a temporary list of (audio_array, label) by applying 
    heavy augmentations to existing data to simulate 'unseen' conditions.
    """
    print(f"\n--- 🧪 Generating Pseudo-Dataset ({num_samples} samples) ---")
    files = [f for f in os.listdir(source_dir) if f.endswith(".wav")]
    
    if not files:
        raise ValueError("Source directory is empty!")

    # Labels map
    labels_map = {'standard': 0, 'kathiawadi': 1, 'saurashtra': 1} # Merge saurashtra into kathiawadi
    
    dataset = []
    augmentor = AudioAugmentor()
    
    # Random selection
    selected_files = random.choices(files, k=num_samples)
    
    for f in tqdm(selected_files, desc="Augmenting Data"):
        # 1. Parse Label
        label_str = f.split('_')[0]
        if label_str not in labels_map:
            continue
        true_label = labels_map[label_str]
        
        # 2. Load
        path = os.path.join(source_dir, f)
        y, sr = librosa.load(path, sr=16000)
        
        # 3. Augment (Create Pseudo-Unseen Data)
        # Apply 1-2 random augmentations
        aug_y = augmentor.augment(y)
        
        # 4. Save to temp file for inference script (since inference expects file path)
        temp_path = f"temp_test_{random.randint(0, 99999)}.wav"
        sf.write(temp_path, aug_y, 16000)
        
        dataset.append((temp_path, true_label, label_str))
        
    return dataset

def run_blind_test():
    print("\n--- 🦁 Running Blind Accuracy Test 🦁 ---")
    
    # 1. Load Model
    predictor = GujaratiDialectPredictor()
    
    # 2. Generate Data
    SOURCE_DIR = "D:/gujarati_project_data/dataset/audio/clean"
    try:
        test_data = create_pseudo_test_set(SOURCE_DIR, num_samples=20) # 20 Random Tests
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # 3. Predict
    y_true = []
    y_pred = []
    
    print(f"\n--- ⚡ Running Inference on {len(test_data)} Samples ---")
    
    # Labels corresponding to 0 and 1 in predictor
    # Check predictor.labels: usually ['Standard', 'Kathiawadi'] where Standard=0, Kathiawadi=1 
    # WAIT! In training:
    # LABELS_MAP = {'standard': 0, 'kathiawadi': 1, 'saurashtra': 1}
    # In predictor: self.labels = ['Standard', 'Kathiawadi']
    # So index 0 is Standard, index 1 is Kathiawadi. Matches.
    
    correct_count = 0
    
    for temp_path, true_label_idx, true_label_str in tqdm(test_data, desc="Testing"):
        try:
            # Predict
            result = predictor.predict(audio_path=temp_path)
            
            # Map prediction text to index
            pred_str = result['final_prediction'].lower()
            if pred_str == 'standard':
                pred_idx = 0
            else:
                pred_idx = 1
            
            y_true.append(true_label_idx)
            y_pred.append(pred_idx)
            
            if pred_idx == true_label_idx:
                correct_count += 1
                
        except Exception as e:
            print(f"Failed on {temp_path}: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 4. Metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Standard', 'Kathiawadi'])
    conf_mat = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"✅ Final Accuracy: {accuracy:.2%}")
    print("="*40)
    print("\n--- Detailed Report ---")
    print(report)
    print("\n--- Confusion Matrix ---")
    print(f"[[TN FP]\n [FN TP]]\n")
    print(conf_mat)
    
    # Interpretation
    print("\n--- Interpretation ---")
    if accuracy > 0.85:
        print("🚀 EXCELLENT RECALL. System is ready for production.")
    elif accuracy > 0.70:
        print("⚠️ GOOD. But augmentations confuse it. Needs more data diversity.")
    else:
        print("❌ POOR. Model is overfitting or augmentations are too strong.")

if __name__ == "__main__":
    run_blind_test()
