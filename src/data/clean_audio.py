
import os
import glob
import webrtcvad
import librosa
import soundfile as sf
import numpy as np

# Config
RAW_AUDIO_DIR = "D:/gujarati_project_data/dataset/audio/raw"
CLEAN_AUDIO_DIR = "D:/gujarati_project_data/dataset/audio/clean"
SAMPLE_RATE = 16000 # VAD requires 16k

def clean_audio_dataset():
    print(f"--- Cleaning Audio Dataset (Gentle VAD) ---")
    os.makedirs(CLEAN_AUDIO_DIR, exist_ok=True)
    
    files = glob.glob(os.path.join(RAW_AUDIO_DIR, "standard_*.wav")) # Focus on Standard first
    # Also include existing ones if needed, but let's prioritize the new batch
    files += glob.glob(os.path.join(RAW_AUDIO_DIR, "saurashtra_*.wav"))
    
    # VAD Mode 1: Less Aggressive (Better for preserving words)
    vad = webrtcvad.Vad(1)
    
    print(f"Found {len(files)} files to process.")
    
    for f in files:
        filename = os.path.basename(f)
        out_path = os.path.join(CLEAN_AUDIO_DIR, filename)
        
        # Skip if already exists (resume)
        if os.path.exists(out_path):
            # print(f"Skipping {filename}, already done.")
            continue

        try:
            # Load and Resample
            y, sr = librosa.load(f, sr=SAMPLE_RATE)
            y = librosa.util.normalize(y)
            
            # Frame setup (30ms)
            frame_duration_ms = 30
            frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)
            
            audio_int16 = (y * 32767).astype(np.int16)
            
            # 1. Detect Flags
            frames = []
            flags = []
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i+frame_size]
                frames.append(frame)
                try:
                    is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
                except:
                    is_speech = False
                flags.append(is_speech)
                
            # 2. Apply Padding (Smoothing)
            # Dilate speech segments by ~300ms (10 frames)
            PADDING = 10
            n_frames = len(flags)
            keep_flags = [False] * n_frames
            
            for i in range(n_frames):
                if flags[i]:
                    start = max(0, i - PADDING)
                    end = min(n_frames, i + PADDING + 1)
                    for j in range(start, end):
                        keep_flags[j] = True
            
            # 3. Collect
            voice_frames = []
            for i in range(n_frames):
                if keep_flags[i]:
                    voice_frames.append(frames[i])
            
            if not voice_frames:
                print(f"Skipping {filename}: No Speech Detected.")
                continue
                
            # Reconstruct
            y_clean = np.concatenate(voice_frames).astype(np.float32) / 32767
            sf.write(out_path, y_clean, SAMPLE_RATE)
            print(f"Processed {filename}: {len(y)/SAMPLE_RATE:.1f}s -> {len(y_clean)/SAMPLE_RATE:.1f}s")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    clean_audio_dataset()
