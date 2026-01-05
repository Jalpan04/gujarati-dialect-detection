# Data Acquisition Strategy: Multimodal Gujarati Dialect Corpus

To train the **Multimodal Fusion Model**, we need "paired" samples: $(Audio, Text, Dialect_Label)$.
Since no such public dataset exists for Gujarati dialects (Saurashtra, Surti, etc.), we must construct one.

## Strategy 1: The "Miner" (YouTube + Alignment)
*Best for: Large-scale, noisy data.*

**Concept**: 
Scrape YouTube videos that naturally contain dialect speech (e.g., Folk events, Dayro, Regional News) and use their auto-generated or manual captions.

**Implementation Steps**:
1.  **Source Identification**:
    *   *Saurashtra*: Search for "Mayabhai Ahir", "Sairam Dave" (Dayro artists), "Saurashtra News".
    *   *Surti*: Search for "Surti Natak", "Surti Comedy".
2.  **Download**: Use `yt-dlp` to download Audio (`.wav`) and Captions (`.srt`).
3.  **Forced Alignment**:
    *   The timestamps in `.srt` files are often loose.
    *   Use a library like `Aeneas` or `Montreal Forced Aligner` to align the audio clips exactly to the text sentences.
4.  **Labeling**: 
    *   Assume the entire video has the dialect of the speaker (Weak Supervision).

**Pros**: Huge volume of data. Real-world noise.
**Cons**: Alignment errors. "Standard" Gujarati captions often mismatched with "Dialect" speech (e.g., Speaker says "Halya" (Saurashtra), Caption says "Chalya" (Standard)).

---

## Strategy 2: The "Synthesizer" (ASR Back-Translation)
*Best for: Leveraging our unlabelled Audio.*

**Concept**:
We have Audio (Phase 2) but no Text. We generate the text using a standard ASR engine.

**Implementation Steps**:
1.  Take raw dialect audio collected in Phase 2.
2.  Pass it through a standard **Gujarati Wav2Vec2** or **Google ASR**.
3.  **The "Error" Feature**: 
    *   The ASR will likely *mis-transcribe* dialect words into Standard Gujarati words.
    *   *Example*: Audio = "Aavshe" (Come), ASR might transcribe "Aashe" (Hope) or just "Aavshe" (if robust).
    *   We use this (Audio, ASR_Text) pair. The model learns that when Audio sounds like $X$ but Text reads like $Y$, it's Dialect $Z$.

**Pros**: Fully automated. No manual transcription needed.
**Cons**: The text is "Standardized" or noisy, not the true dialect text.

---

## Strategy 3: The "Crowd" (Active Collection)
*Best for: High-quality, Gold Standard evaluations.*

**Concept**:
Build a simple UI (Streamlit/Web) to ask users to record sentences.

**Implementation Steps**:
1.  Take 100 sentences from our **Wikisource Text Corpus** (Phase 1).
2.  Show a sentence to a user: "Please read this in your native accent."
3.  User records audio.
4.  Result: Perfect `(Text, Audio, Dialect)` tuple.

**Pros**: Highest quality.
**Cons**: Requires human volume (expensive/slow).

---

## Recommended Roadmap for MVP

1.  **Immediate**: Use **Strategy 2 (Synthesizer)**. 
    *   We assume we will get some audio files. 
    *   We can run them through a free ASR library to get the "Text" component.
    *   This allows us to train the Fusion pipeline.
2.  **Future**: Build the **YouTube Miner** to scale up.
