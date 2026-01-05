
# 🦁 Gujarati Dialect Intelligence System - Final Version

## 🎯 Project Overview
This system identifies **Gujarati Dialects** from both **Text** and **Audio**.
It has been optimized to distinguish between:
1.  **Standard Gujarati** (Official/Literary)
2.  **Kathiawadi** (Saurashtra Region/Folk)

*Note: Surti/Kutchi were removed to ensure high accuracy on the available data.*

## 🚀 How to Run
### 1. Start the UI (Easiest)
Run this command in VS Code terminal:
```powershell
streamlit run src/app.py
```
Then open: **[http://localhost:8501](http://localhost:8501)**

### 2. Command Line (For Testing)
```powershell
# Text
python src/inference.py --text "તમે ક્યાં જાવ છો?"

# Audio
python src/inference.py --audio "path/to/my_recording.wav"
```

## 🛠️ System Components
-   **Text Model**: `models/baseline_svm.pkl` (Calibrated SVM, F1 ~83%)
-   **Audio Model**: `models/audio_crnn.pth` (2-Class CRNN, Loss ~0.005)
-   **Data**: Stored in `D:/gujarati_project_data/`

## ❓ Troubleshooting
-   **"Old Results" in Audio?**
    -   Just click "Analyze" again. The system now forces a fresh read of the file.
-   **Borderline Predictions?**
    -   Poetic/Classic text often looks like "Standard" to the AI.
    -   Strong colloquialisms ("કાં બાપા") trigger "Kathiawadi".

## 👨‍💻 Developed By
**Antigravity Agent** & **User**
