
import os
import glob
import subprocess
import shutil

# Config
RAW_AUDIO_DIR = "D:/gujarati_project_data/dataset/audio/raw"
SEPARATED_DIR = "D:/gujarati_project_data/dataset/audio/separated"
MODEL = "htdemucs" # Fast and good (comes with Demucs v4)

def separate_vocals():
    print(f"--- Separating Vocals (Demucs) ---")
    os.makedirs(SEPARATED_DIR, exist_ok=True)
    
    # We process RAW files because 'clean' files might have already cut words badly
    # We want to go back to source -> remove music -> then VAD
    files = glob.glob(os.path.join(RAW_AUDIO_DIR, "*.wav"))
    
    print(f"Found {len(files)} files to separate.")
    
    for f in files:
        filename = os.path.basename(f)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Check if already done
        expected_output = os.path.join(SEPARATED_DIR, MODEL, name_no_ext, "vocals.wav")
        if os.path.exists(expected_output):
             # print(f"Skipping {filename}, vocals already extracted.")
             continue
             
        print(f"Processing: {filename} ...")
        
        # Run Demucs Command Line
        # demucs -n htdemucs -o <OUT> <FILE>
        cmd = [
            "demucs",
            "-n", MODEL,
            "-o", SEPARATED_DIR,
            "--two-stems", "vocals", # Only save 'vocals' and 'no_vocals' (saves disk/time)
            f
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Separated {filename}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error separating {filename}: {e}")

    print("\n--- Separation Complete ---")
    print(f"Vocal tracks are in: {SEPARATED_DIR}/{MODEL}/<song_name>/vocals.wav")
    print("Run clean_audio.py targeting THIS folder next.")

if __name__ == "__main__":
    separate_vocals()
