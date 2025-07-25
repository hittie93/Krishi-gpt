import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dotenv
from typing import List, Tuple, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.schema import Document

from backend.translate import translate_to_english, translate_from_english, detect_language
from backend.tts_response import speak_response

# (optional) for nicer error handling of Google GenAI
try:
    from google.api_core.exceptions import ServiceUnavailable
except Exception:  # pragma: no cover
    ServiceUnavailable = Exception

dotenv.load_dotenv()

# === Configuration ===
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set in .env file")
if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY not set. Groq fallback will be disabled.")

# === Embeddings ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# === FAISS Index Path ===
FAISS_INDEX_PATH = "data/faiss_index"

def build_faiss_index():
    """Build FAISS index if not found."""
    print("⚠️ FAISS index not found. Building a new one...")
    docs_folder = "data/cleaned_docs"
    documents = []

    for file in os.listdir(docs_folder):
        if file.endswith(".txt"):
            with open(os.path.join(docs_folder, file), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append(Document(page_content=content))

    if not documents:
        raise FileNotFoundError("❌ No documents found in data/cleaned_docs to build FAISS index.")

    vectorstore = FAISS.from_documents(documents, embedding_model)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index built successfully.")

# === Load or Build Vector Store ===
if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
    build_faiss_index()

vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# === LLMs ===
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
    temperature=0.3,
)

groq_llm = None
if GROQ_API_KEY:
    groq_llm = ChatGroq(
        model="llama3-70b-8192",
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
    )

# === Prompts ===
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are KrishiGPT, an agriculture assistant. "
        "Answer the question **only** using the provided context. If the context "
        "doesn't contain the answer, say so clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)

OPEN_WEB_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are KrishiGPT, an agriculture expert. The local knowledge base "
        "does not contain the answer. Provide a concise, accurate answer from your "
        "general knowledge. If discussing Indian government schemes, clearly name "
        "the scheme, eligibility, benefits, and application steps.\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)

# ---------- Core helpers ----------

def _is_unhelpful_answer(text: str) -> bool:
    """Detects useless answers from FAISS-based RAG."""
    if not text:
        return True
    text_lower = text.lower().strip()
    return (
        "does not contain" in text_lower
        or "no information available" in text_lower
        or "i don't know" in text_lower
    )

def _call_llm_with_rag(llm, question_en: str) -> Tuple[str, List[Document]]:
    """Retrieve context and answer with RAG. Returns (answer_en, docs)."""
    docs = retriever.invoke(question_en)
    context = "\n".join(d.page_content for d in docs if d.page_content.strip())
    if not context.strip():
        return "", docs  # signal empty context
    prompt = RAG_PROMPT.format(context=context, question=question_en)
    resp = llm.invoke(prompt)
    text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    return text, docs

def _call_llm_open(llm, question_en: str) -> str:
    """Direct LLM call without RAG (fallback)."""
    prompt = OPEN_WEB_PROMPT.format(question=question_en)
    resp = llm.invoke(prompt)
    return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

def _answer_with_fallback(question_en: str) -> Tuple[str, List[Document], bool]:
    """
    Try: RAG with Gemini → (if empty context/unhelpful) direct Gemini →
    (if fails / quota) Groq (RAG first, then open).
    """
    used_open_fallback = False
    docs: List[Document] = []

    try:
        answer_en, docs = _call_llm_with_rag(gemini_llm, question_en)
        if _is_unhelpful_answer(answer_en):
            print("⚠️ No relevant info in documents. Falling back to Gemini (open).")
            used_open_fallback = True
            answer_en = _call_llm_open(gemini_llm, question_en)
        return answer_en, docs, used_open_fallback

    except (ServiceUnavailable, Exception) as e:
        err = str(e).lower()
        if groq_llm and ("429" in err or "quota" in err or "serviceunavailable" in err or "503" in err):
            print("⚠️ Gemini unavailable/quota exceeded. Falling back to Groq (Llama3-70B).")
            try:
                answer_en, docs = _call_llm_with_rag(groq_llm, question_en)
                if _is_unhelpful_answer(answer_en):
                    print("⚠️ No context with Groq. Using Groq (open).")
                    used_open_fallback = True
                    answer_en = _call_llm_open(groq_llm, question_en)
                return answer_en, docs, used_open_fallback
            except Exception as e2:
                raise RuntimeError(f"Groq fallback also failed: {e2}") from e2
        raise

# ---------- Public API ----------
def ask_question(query: str):
    print(f"🔍 Input Query: {query}")

    detected_lang = detect_language(query)
    print(f"🌐 Detected Language: {detected_lang}")

    translated_query = translate_to_english(query) if detected_lang != 'en' else query
    print(f"🌐 Translated Query (EN): {translated_query}")

    # Get answer with fallback
    answer_en, _, _ = _answer_with_fallback(translated_query)
    print(f"✅ Answer (EN): {answer_en}")

    final_answer = translate_from_english(answer_en, query) if detected_lang != 'en' else answer_en
    print(f"🌐 Final Answer: {final_answer}")

    try:
        speak_response(final_answer, lang=detected_lang)
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")

    return final_answer

def answer_query_for_ui(query: str, top_k: int = 3, speak: bool = False):
    lang = detect_language(query)
    q_en = translate_to_english(query) if lang != "en" else query

    docs = retriever.invoke(q_en)[:top_k]
    answer_en, _, used_open = _answer_with_fallback(q_en)
    final = translate_from_english(answer_en, query) if lang != "en" else answer_en

    audio_path: Optional[str] = None
    if speak:
        try:
            audio_path = speak_response(final, lang=lang)
        except Exception as e:
            print(f"🔇 TTS failed (continuing without voice): {e}")

    return {
        "lang": lang,
        "answer_en": answer_en,
        "answer": final,
        "contexts": [d.page_content for d in docs],
        "used_open_fallback": used_open,
        "audio_path": audio_path,
    }

# ---------- CLI loop ----------
if __name__ == "__main__":
    while True:
        q = input("\n❓ Ask your farming question: ")
        if q.lower() in {"exit", "quit"}:
            break
        try:
            resp = ask_question(q)
            print("\n💬 Response:", resp)
        except Exception as e:
            print(f"❌ Error: {e}")
