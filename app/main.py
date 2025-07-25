# app/main.py
import sys, os, io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import dotenv
dotenv.load_dotenv()

from backend.rag_pipeline import answer_query_for_ui
from backend.tts_response import text_to_speech  # for UI playback


# ------- Optional voice libs (handled gracefully) -------
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
# -------------------------------------------------------

st.set_page_config(page_title="KrishiGPT 🌾", page_icon="🌾", layout="wide")

st.title("🌾 KrishiGPT – किसान का AI साथी")
st.caption("Hindi / English • RAG + Gemini/Groq fallback • FAISS • TTS • Voice input")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("⚙️ Settings")
    input_mode = st.radio("Input mode", ["📝 Text", "🎤 Voice"], index=0)
    play_audio = st.checkbox("🔊 Speak answer", value=False)
    show_context = st.checkbox("📚 Show retrieved context", value=True)
    top_k = st.slider("Top context chunks to display", 1, 10, 3)
st.write("---")

# ---------------- Voice helpers ----------------
def transcribe_audio_bytes(audio_bytes: bytes, language_hint: str = "hi-IN") -> str:
    """Transcribe WAV/MP3 bytes to text using SpeechRecognition (Google Web API)."""
    if not HAS_SR:
        st.error("SpeechRecognition is not installed. Run: `pip install SpeechRecognition`")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        # try Hindi first, then English
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
    st.markdown("#### अपना सवाल पूछें / Ask your question")
    user_q = st.text_area(
        "Type in Hindi or English",
        placeholder="उदाहरण: धान की फसल के लिए सबसे अच्छा उर्वरक क्या है?\nExample: What is the best fertilizer for rice crops?",
        height=120
    )

    if st.button("🤖 Get Answer"):
        run_pipeline(user_q)

# ---------------- UI: Voice mode ----------------
def voice_mode_ui():
    st.markdown("#### 🎤 बोलकर पूछें / Ask by speaking")

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
        st.warning("`streamlit-mic-recorder` not installed or not supported. Falling back to file upload.")
        uploaded = st.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3"])
        if uploaded:
            with st.spinner("Transcribing..."):
                transcribed_text = transcribe_audio_bytes(uploaded.read())
                if transcribed_text:
                    st.success(f"🗣️ You said: **{transcribed_text}**")

    # If we got text from mic/upload, show "Get Answer"
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

    # Show answer
    st.markdown("### ✅ Answer")
    st.write(result["answer"])

    # Optional audio
    if play_audio:
        fp = text_to_speech(
            result["answer"],
            lang="hi" if result["lang"] == "hi" else "en",
            autoplay=False
        )
        if fp and os.path.exists(fp):
            with open(fp, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
        else:
            st.info("Audio could not be generated.")

    # Show English version as debug (optional)
    with st.expander("🔤 English debug answer"):
        st.write(result["answer_en"])

    # Show contexts
    if show_context and result.get("contexts"):
        st.markdown("### 📚 Retrieved context")
        for i, ctx in enumerate(result["contexts"], 1):
            with st.expander(f"Chunk #{i}"):
                st.write(ctx)

# --------------- Entry point -----------------
if input_mode == "📝 Text":
    text_mode_ui()
else:
    voice_mode_ui()

st.markdown("---")
st.caption("Built with ❤ for Indian farmers — KrishiGPT")
