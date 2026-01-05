import os
import yt_dlp
import pandas as pd

# Configuration
OUTPUT_DIR = "D:/gujarati_project_data/dataset/audio/raw"
MAX_DOWNLOADS_PER_QUERY = 20 # Production: 20 videos per query (~1-2 GB potential)

# Dialect-Specific Queries
# These queries target specific cultural artifacts associated with dialects.
QUERIES = {
    "saurashtra": [
        "Zaverchand Meghani Lok Sahitya Audio",
        "Sairam Dave Dayro",
        "Mayabhai Ahir Jokes",
        "Kathiawadi Dayro"
    ],
    "surti": [
        "Surti Funny Video",
        "Surti Natak Audio",
         # Specific Surti YouTube channels or content
    ],
    "standard": [
        "Gujarati News Audio",
        "Gujarati Audiobook",
        "Gujarat Samachar News"
    ]
}

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def download_audio(dialect, query):
    """
    Downloads audio from YouTube searches.
    """
    print(f"Searching for [{dialect}]: {query}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': f'{OUTPUT_DIR}/{dialect}_%(id)s.%(ext)s',
        'noplaylist': True,
        'playlistend': MAX_DOWNLOADS_PER_QUERY,
        'quiet': True,
        'ignoreerrors': True,
        'download_archive': os.path.join(OUTPUT_DIR, 'download_history.txt'), # Skip already downloaded
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # "ytsearchN:" searches and downloads top N results
            search_query = f"ytsearch{MAX_DOWNLOADS_PER_QUERY}:{query}"
            ydl.download([search_query])
    except Exception as e:
        print(f"Error downloading {query}: {e}")

def main():
    setup_directories()
    
    for dialect, queries in QUERIES.items():
        print(f"\n--- Mining Dialect: {dialect.upper()} ---")
        for q in queries:
            download_audio(dialect, q)
            
    print("\n[COMPLETE] Download finished. Check dataset/audio/raw/")

if __name__ == "__main__":
    main()
