import os
import yt_dlp

# V2: Targeted "Standard Gujarati" Mining
# Focus: News, Audiobooks, Interviews
# Goal: 100 Hours of Speech | No Surti

OUTPUT_DIR = "D:/gujarati_project_data/dataset/audio/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sources: High Quality Standard Gujarati Speech
SEARCH_QUERIES = [
    # News Channels (Pure Standard)
    "sandesh news gujarati full bulletin",
    "gstv gujarati samachar full",
    "vtv gujarati news debate",
    "abp asmita full episode",
    
    # Audiobooks / Literature (High Fidelity)
    "gujarati audiobook full",
    "meghani sahitya audiobook", # Might contain Kathiawadi, need filter later, but generally formal
    "gujarati sahitya akademi speech",
    "gujarati motivational speech"
]

def download_audio(query, limit=10):
    print(f"--- Mining: {query} ---")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': f"{OUTPUT_DIR}/standard_{query.split()[0]}_%(id)s.%(ext)s",
        'quiet': False,
        'ignoreerrors': True,
        'noplaylist': True,
        'max_downloads': limit,
        # Duration filter: Avoid shorts (<60s) and too long (>20m to save time for now)
        'match_filter': yt_dlp.utils.match_filter_func("duration > 60 & duration < 1200") 
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([f"ytsearch{limit}:{query}"])
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Phase 3: Standard Gujarati Crusade...")
    
    # Run targeted searches
    for q in SEARCH_QUERIES:
        download_audio(q, limit=5) # 5 per query = ~40 new files initially
        
    print("\n✅ Mining Complete. Now run 'clean_audio.py' to VAD filter them.")
