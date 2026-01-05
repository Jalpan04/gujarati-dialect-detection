
import os
import sys
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import GujaratiDialectPredictor

def validate_system():
    print("--- 🦁 Final System Validation 🦁 ---")
    
    # 1. Initialize
    print("1. Loading Predictor...")
    try:
        predictor = GujaratiDialectPredictor()
        print("   ✅ Loaded successfully.")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return

    # 2. Test Audio
    audio_dir = "D:/gujarati_project_data/dataset/audio/clean"
    if os.path.exists(audio_dir):
        files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
        if files:
            test_file = os.path.join(audio_dir, random.choice(files))
            print(f"\n2. Testing Audio Inference on: {os.path.basename(test_file)}")
            try:
                result = predictor.predict(audio_path=test_file)
                print(f"   ✅ Prediction: {result}")
            except Exception as e:
                print(f"   ❌ Audio Inference Failed: {e}")
        else:
             print("   ⚠️ No audio files found to test.")
    else:
        print("   ⚠️ Audio directory not found.")
        
    print("\n--- Validation Complete ---")

if __name__ == "__main__":
    validate_system()
