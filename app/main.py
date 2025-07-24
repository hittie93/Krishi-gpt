# app/main.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from backend.rag_pipeline import answer_query_for_ui
from backend.tts_response import text_to_speech  # if you want to generate & play in UI

st.set_page_config(page_title="KrishiGPT 🌾", page_icon="🌾", layout="wide")

st.title("🌾 KrishiGPT – किसान का AI साथी")
st.caption("Hindi / English • RAG + Gemini • FAISS • TTS")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    play_audio = st.checkbox("🔊 Speak answer", value=False)
    show_context = st.checkbox("📚 Show retrieved context", value=True)
    top_k = st.slider("Top context chunks to display", 1, 10, 3)

st.markdown("#### अपना सवाल पूछें / Ask your question")
user_q = st.text_area(
    "Type in Hindi or English",
    placeholder="उदाहरण: धान की फसल के लिए सबसे अच्छा उर्वरक क्या है?\nExample: What is the best fertilizer for rice crops?",
    height=120
)

if st.button("🤖 Get Answer"):
    if not user_q.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Thinking..."):
        try:
            result = answer_query_for_ui(user_q, top_k=top_k, speak=False)
        except Exception as e:
            st.error(f"❌ Failed to answer: {e}")
            st.stop()

    # Show answer
    st.markdown("### ✅ Answer")
    st.write(result["answer"])

    # Optional audio
    if play_audio:
        # Generate an mp3 and play it
        from backend.tts_response import text_to_speech
        fp = text_to_speech(result["answer"], lang="hi" if result["lang"] == "hi" else "en", autoplay=False)
        if fp and os.path.exists(fp):
            with open(fp, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
        else:
            st.info("Audio could not be generated.")

    # Show English version as debug (optional)
    with st.expander("🔤 English debug answer"):
        st.write(result["answer_en"])

    # Show contexts
    if show_context and result["contexts"]:
        st.markdown("### 📚 Retrieved context")
        for i, ctx in enumerate(result["contexts"], 1):
            with st.expander(f"Chunk #{i}"):
                st.write(ctx)

st.markdown("---")
st.caption("Built with ❤ for Indian farmers — KrishiGPT")
