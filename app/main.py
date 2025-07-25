# app/main.py
import sys, os, io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import dotenv
dotenv.load_dotenv()

from backend.rag_pipeline import answer_query_for_ui
from backend.tts_response import text_to_speech  # for UI playback

# ------- Optional voice libs -------
HAS_MIC = False
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    pass

HAS_SR = False
try:
    import speech_recognition as sr
    HAS_SR = True
except Exception:
    pass
# -----------------------------------

st.set_page_config(page_title="KrishiGPT 🌾", page_icon="🌾", layout="wide")

# --------- Advanced Glassmorphism CSS ---------
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
            background-size: 400% 400%;
            animation: gradientBG 10s ease infinite;
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }
        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .main {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 30px;
            margin-top: 20px;
        }
        h1, h2, h3 {
            color: #00ffc6;
            text-shadow: 0px 0px 10px #00ffc6, 0px 0px 20px #00ffc6;
        }
        .stButton>button {
            background: linear-gradient(135deg, #00ffc6, #00bcd4);
            border: none;
            border-radius: 10px;
            color: #0b0c10;
            font-weight: bold;
            padding: 12px 24px;
            font-size: 18px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #00bcd4, #00ffc6);
            box-shadow: 0 0 15px #00ffc6;
            color: black;
        }
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 8px;
            border: 1px solid #00ffc6;
        }
        .stRadio>div {
            color: white;
            font-weight: bold;
        }
        .stSlider {
            color: white;
        }
        .css-1d391kg, .css-1kyxreq, .css-1g6gooi {
            background-color: transparent !important;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --------- Header ---------
st.title("🌾 KrishiGPT – किसान का AI साथी")
st.caption("Hindi / English • RAG + Gemini/Groq fallback • FAISS • TTS • Voice input")

# --------- Sidebar ---------
with st.sidebar:
    st.subheader("⚙️ Settings")
    input_mode = st.radio("Input mode", ["📝 Text", "🎤 Voice"], index=0)
    play_audio = st.checkbox("🔊 Speak answer", value=False)
    show_context = st.checkbox("📚 Show retrieved context", value=True)
    top_k = st.slider("Top context chunks to display", 1, 10, 3)
st.write("---")

# ---------------- Voice helpers ----------------
def transcribe_audio_bytes(audio_bytes: bytes, language_hint: str = "hi-IN") -> str:
    if not HAS_SR:
        st.error("SpeechRecognition is not installed. Run: `pip install SpeechRecognition`")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        for lang in [language_hint, "en-IN", "en-US", "hi-IN"]:
            try:
                txt = recognizer.recognize_google(audio, language=lang)
                if txt.strip():
                    return txt
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                st.error(f"Speech API error: {e}")
                break
    except Exception as e:
        st.error(f"Could not process audio: {e}")
    return ""

# ---------------- UI: Text mode ----------------
def text_mode_ui():
    st.markdown("### अपना सवाल पूछें / Ask your question")
    user_q = st.text_area(
        "Type in Hindi or English",
        placeholder="उदाहरण: धान की फसल के लिए सबसे अच्छा उर्वरक क्या है?\nExample: What is the best fertilizer for rice crops?",
        height=120
    )
    if st.button("🤖 Get Answer"):
        run_pipeline(user_q)

# ---------------- UI: Voice mode ----------------
def voice_mode_ui():
    st.markdown("### 🎤 बोलकर पूछें / Ask by speaking")
    transcribed_text = ""
    if HAS_MIC:
        st.info("Click the mic, speak, and then click again to stop.")
        audio_dict = mic_recorder(
            start_prompt="🎙️ Start recording",
            stop_prompt="🛑 Stop",
            just_once=False,
            key="mic",
        )
        if audio_dict and "bytes" in audio_dict and audio_dict["bytes"] is not None:
            with st.spinner("Transcribing..."):
                transcribed_text = transcribe_audio_bytes(audio_dict["bytes"])
                if transcribed_text:
                    st.success(f"🗣️ You said: **{transcribed_text}**")
    else:
        st.warning("`streamlit-mic-recorder` not installed. Upload audio instead.")
        uploaded = st.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3"])
        if uploaded:
            with st.spinner("Transcribing..."):
                transcribed_text = transcribe_audio_bytes(uploaded.read())
                if transcribed_text:
                    st.success(f"🗣️ You said: **{transcribed_text}**")

    if transcribed_text:
        if st.button("🤖 Get Answer (from voice)"):
            run_pipeline(transcribed_text)

# ---------------- Core runner ----------------
def run_pipeline(query: str):
    if not query or not query.strip():
        st.warning("Please provide a valid question (text or voice).")
        return

    with st.spinner("Thinking..."):
        try:
            result = answer_query_for_ui(query, top_k=top_k, speak=False)
        except Exception as e:
            st.error(f"❌ Failed to answer: {e}")
            return

    st.markdown("## ✅ Answer")
    st.write(result["answer"])

    if play_audio:
        fp = text_to_speech(result["answer"], lang="hi" if result["lang"] == "hi" else "en", autoplay=False)
        if fp and os.path.exists(fp):
            with open(fp, "rb") as f:
                st.audio(f.read(), format="audio/mp3")

    with st.expander("🔤 English debug answer"):
        st.write(result["answer_en"])

    if show_context and result.get("contexts"):
        st.markdown("## 📚 Retrieved context")
        for i, ctx in enumerate(result["contexts"], 1):
            with st.expander(f"Chunk #{i}"):
                st.write(ctx)

# -------- Entry Point --------
if input_mode == "📝 Text":
    text_mode_ui()
else:
    voice_mode_ui()

st.markdown("---")
st.caption("Built with ❤ for Indian farmers — KrishiGPT")
