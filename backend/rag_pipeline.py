import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from backend.translate import translate_to_english, translate_from_english, detect_language
from backend.tts_response import speak_response
import dotenv

dotenv.load_dotenv()

# === Configuration ===
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set in .env file")

# === Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
                content = f.read()
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

vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# === Gemini LLM Setup ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Change here
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
    temperature=0.3
)


# === Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Answer the question based on the provided context:

        Context:
        {context}

        Question:
        {question}

        Answer:
    """
)

# === RAG Chain ===
rqa_chain = (
    RunnableMap({
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    })
    | RunnableMap({
        "context": lambda x: "\n".join([doc.page_content for doc in x["context"]]),
        "question": lambda x: x["question"]
    })
    | prompt_template
    | llm
)

def ask_question(query: str):
    print(f"🔍 Input Query: {query}")

    detected_lang = detect_language(query)
    print(f"🌐 Detected Language: {detected_lang}")

    translated_query = translate_to_english(query) if detected_lang != 'en' else query
    print(f"🌐 Translated Query (EN): {translated_query}")

    answer = rqa_chain.invoke({"question": translated_query})
    final_answer = answer.content.strip() if hasattr(answer, 'content') else str(answer).strip()

    print(f"✅ Answer (EN): {final_answer}")

    if detected_lang != 'en':
        final_answer = translate_from_english(final_answer, query)
        print(f"🌐 Translated Back Answer: {final_answer}")

    speak_response(final_answer, lang=detected_lang)
    return final_answer

def answer_query_for_ui(query: str, top_k: int = 3, speak: bool = False):
    """
    UI-friendly wrapper that:
    - detects language
    - translates to EN, runs RAG
    - translates back to original language
    - (optionally) does TTS
    - returns structured response + top contexts
    """
    detected_lang = detect_language(query)
    translated_query = translate_to_english(query) if detected_lang != "en" else query

    # retrieve top_k docs for UI display
    docs = retriever.get_relevant_documents(translated_query)[:top_k]
    joined_ctx = "\n".join([d.page_content for d in docs])

    # ask LLM
    answer = rqa_chain.invoke({"question": translated_query})
    answer_en = answer.content.strip() if hasattr(answer, "content") else str(answer).strip()

    final_answer = (
        translate_from_english(answer_en, query) if detected_lang != "en" else answer_en
    )

    audio_path = None
    if speak:
        audio_path = speak_response(final_answer, lang=detected_lang)

    return {
        "lang": detected_lang,
        "answer_en": answer_en,
        "answer": final_answer,
        "contexts": [d.page_content for d in docs],
        "audio_path": audio_path,
    }

if __name__ == "__main__":
    while True:
        q = input("\n❓ Ask your farming question: ")
        if q.lower() in ["exit", "quit"]:
            break
        response = ask_question(q)
        print("\n💬 Response:", response)
