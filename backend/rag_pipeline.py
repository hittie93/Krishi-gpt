import sys
import os
import requests

# Define project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import dotenv
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.schema import Document

from backend.translate import detect_language, translate_to_english, translate_from_english
from backend.tts_response import speak_response

# Load .env from root
dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# === API KEYS ===
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Add this in .env

if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set in .env file")

# === Embeddings & Vector DB ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = "data/faiss_index"

def build_faiss_index():
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
        raise FileNotFoundError("❌ No documents found in data/cleaned_docs.")

    vectorstore = FAISS.from_documents(documents, embedding_model)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index built successfully.")

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
    temperature=0.3,
)

groq_llm = ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,
) if GROQ_API_KEY else None

# === Prompts ===
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are KrishiGPT, an agriculture assistant. Answer the question "
        "ONLY using the provided context. Do not answer unrelated questions.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    ),
)

OPEN_WEB_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are KrishiGPT, an agriculture expert for Indian farmers. "
        "Answer the question with detailed, practical farming advice, including soil type, "
        "climate, irrigation, fertilizers, and best practices if applicable. "
        "Focus ONLY on crops, farming, fertilizers, soil, pesticides, weather, or Indian govt schemes. "
        "If the question is unrelated (e.g., sports, movies), reply: "
        "'I can only assist with agriculture-related queries.'\n\n"
        "Question:\n{question}\nAnswer:"
    ),
)

# === Weather Integration ===
def get_weather_info(location: str) -> str:
    if not WEATHER_API_KEY:
        return "Weather API key not set."
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return "Weather data not found for this location."
        main = response['main']
        weather = response['weather'][0]['description']
        temp = main['temp']
        feels_like = main['feels_like']
        return f"Current weather in {location.title()}: {weather}, Temp: {temp}°C (Feels like {feels_like}°C)."
    except Exception as e:
        return f"Error fetching weather: {e}"

def get_weather_with_crop_advice(location: str, crop_query: str) -> str:
    weather_info = get_weather_info(location)
    if "Weather data not found" in weather_info or "Error" in weather_info:
        return weather_info

    analysis_prompt = (
        f"{weather_info}\n\nBased on this weather, provide crop precautions, "
        f"pest control tips, and recommendations for farmers. Query: {crop_query}"
    )
    try:
        resp = gemini_llm.invoke(analysis_prompt)
        return weather_info + "\n\nAgriculture Advice: " + resp.content.strip()
    except Exception:
        return weather_info

def detect_weather_query(query: str) -> Optional[str]:
    keywords = ["weather", "temperature", "rain", "forecast", "मौसम", "बारिश", "तापमान"]
    if any(kw.lower() in query.lower() for kw in keywords):
        words = query.split()
        for i, w in enumerate(words):
            if w.lower() in ["in", "at", "of"] and i + 1 < len(words):
                return words[i + 1]
        return "Delhi"
    return None

# === Core Answer Logic ===
def _call_llm_with_rag(llm, question: str) -> Tuple[str, List[Document]]:
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    if not context.strip():
        return "", docs
    prompt = RAG_PROMPT.format(context=context, question=question)
    resp = llm.invoke(prompt)
    return resp.content.strip(), docs

def _call_llm_open(llm, question: str) -> str:
    prompt = OPEN_WEB_PROMPT.format(question=question)
    resp = llm.invoke(prompt)
    return resp.content.strip()

def _answer_with_fallback(question: str) -> Tuple[str, List[Document]]:
    docs = []
    try:
        # Step 1: Try with RAG
        answer, docs = _call_llm_with_rag(gemini_llm, question)

        # Step 2: Check if RAG answer is invalid or not helpful
        if (not answer or len(answer) < 10 or
            "cannot be answered" in answer.lower() or
            "no information" in answer.lower() or
            "I can only assist" in answer):
            print("⚠️ RAG did not give a valid answer. Switching to open-domain Gemini.")
            answer = _call_llm_open(gemini_llm, question)

        return answer, docs

    except Exception as e:
        print(f"⚠️ Error in Gemini RAG: {e}")
        if groq_llm:
            print("⚠️ Switching to Groq fallback.")
            answer, docs = _call_llm_with_rag(groq_llm, question)
            if not answer or len(answer) < 10:
                answer = _call_llm_open(groq_llm, question)
            return answer, docs
        raise e


# === Public APIs ===
def ask_question(query: str):
    print(f"🔍 Query: {query}")
    lang = detect_language(query)

    location = detect_weather_query(query)
    if location:
        answer = get_weather_with_crop_advice(location, query)
        return translate_from_english(answer, query) if lang != "en" else answer

    answer, _ = _answer_with_fallback(query)
    return translate_from_english(answer, query) if lang != "en" else answer

def answer_query_for_ui(query: str, top_k: int = 3, speak: bool = False):
    lang = detect_language(query)

    location = detect_weather_query(query)
    if location:
        final = get_weather_with_crop_advice(location, query)
        final = translate_from_english(final, query) if lang != "en" else final
        return {
            "lang": lang,
            "answer_en": final if lang == "en" else translate_to_english(final),
            "answer": final,
            "contexts": [],
            "used_open_fallback": False,
            "audio_path": None,
        }

    docs = retriever.invoke(query)[:top_k]
    answer, _ = _answer_with_fallback(query)
    final = translate_from_english(answer, query) if lang != "en" else answer

    return {
        "lang": lang,
        "answer_en": answer,
        "answer": final,
        "contexts": [d.page_content for d in docs],
        "used_open_fallback": False,
        "audio_path": None,
    }
