import streamlit as st
import os
import sys
import tempfile
import numpy as np
import random
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import GujaratiDialectPredictor

# Page Config
st.set_page_config(
    page_title="Gujarati Dialect AI (Phase 2)",
    page_icon="🦁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .success-text { color: green; font-weight: bold; }
    .warning-text { color: orange; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 1. Load System (Cached)
@st.cache_resource
def load_system():
    # Use the unified Inference Class which handles MuRIL + CRNN
    return GujaratiDialectPredictor()

try:
    with st.spinner("Loading AI Brain (Phase 2: MuRIL + Clean CRNN)..."):
        predictor = load_system()
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# Sidebar
st.sidebar.title("🦁 Gujarati AI")
st.sidebar.info("**Phase 2**: Transformer Edition")
st.sidebar.markdown("---")
st.sidebar.write("### Model Status")
st.sidebar.success("✅ Text: MuRIL (Google)")
st.sidebar.success("✅ Audio: Wav2Vec2 (Meta)")

# Main Interface
st.markdown("<h1 style='text-align: center;'>🦁 Gujarati Dialect Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by <b>Google MuRIL</b> & <b>VAD-Cleaned Audio</b></p>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📝 Text Analysis", "🎤 Audio Analysis"])

# --- TEXT TAB ---
with tab1:
    col1, col2 = st.columns([3, 1])
    
    sample_texts = {
        'kathiawadi': [
            "કાં ભાઈ, ક્યાં હાલ્યા ? મજા માં ને ?",
            "એલા ભાઈ, તું તો સાવ ગાંડો છે હો.",
            "આમ જો, ઓલી ભેંસ ભાગી ગઈ."
        ],
        'standard': [
            "તમે ક્યાં જઈ રહ્યા છો? બધું કુશળ છે ને?",
            "આજે હવામાન ખૂબ જ સુંદર છે.",
            "ગુજરાતી ભાષા સાહિત્ય સમૃદ્ધ છે."
        ]
    }
    
    with col2:
        st.write("### 🎲 Samples")
        if st.button("Load Kathiawadi"):
            st.session_state.text_input = random.choice(sample_texts['kathiawadi'])
        if st.button("Load Standard"):
            st.session_state.text_input = random.choice(sample_texts['standard'])
            
    with col1:
        text_input = st.text_area("Enter Gujarati Text", 
                                 value=st.session_state.get('text_input', ""),
                                 height=150)
        
    if st.button("Analyze Text"):
        if text_input:
             result = predictor.predict(text=text_input)
             
             pred = result['final_prediction']
             conf = result['confidence']
             
             st.markdown(f"""
             <div class="prediction-box">
                <h2>{pred}</h2>
                <p>Confidence: {conf:.1%}</p>
             </div>
             """, unsafe_allow_html=True)
             
             # Chart
             if 'details' in result:
                 probs = result['details']
                 # Probs is now a dict {label: probability}
                 st.bar_chart(probs)
        else:
            st.warning("Please enter some text.")

# --- AUDIO TAB ---
with tab2:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.write("### 🎤 Audio Input")
        input_method = st.radio("Source:", ["Microphone", "Upload File"], horizontal=True)
        
        audio_file = None
        if input_method == "Microphone":
            audio_file = st.audio_input("Record Voice")
        else:
            audio_file = st.file_uploader("Upload .wav", type=['wav'])
            
    with col_b:
        st.info("💡 **Tip**: The model is now trained on **speech only**. Avoid singing or background music for best results.")

    if audio_file:
        st.audio(audio_file)
        
        if st.button("Analyze Audio"):
             audio_file.seek(0)
             bytes_data = audio_file.read()
             
             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                 tmp.write(bytes_data)
                 tmp_path = tmp.name
                 
             st.info(f"Processing... (Size: {len(bytes_data)} bytes)")
             
             try:
                 result = predictor.predict(audio_path=tmp_path)
                 
                 pred = result['final_prediction']
                 conf = result['confidence']
                 
                 st.markdown(f"""
                 <div class="prediction-box">
                    <h2>{pred}</h2>
                    <p>Confidence: {conf:.1%}</p>
                 </div>
                 """, unsafe_allow_html=True)
                 
                 if 'details' in result:
                     st.bar_chart({
                         'Standard': result['details'][0],
                         'Kathiawadi': result['details'][1]
                     })
             except Exception as e:
                 st.error(f"Analysis failed: {e}")
             finally:
                 if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
