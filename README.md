
# 🌾 KrishiGPT – किसान का AI साथी  

KrishiGPT is an AI-powered assistant built for **Indian farmers**, providing reliable agricultural guidance in **Hindi and English**. It leverages **Retrieval-Augmented Generation (RAG)** with FAISS vector search, government crop guides, and scheme PDFs to deliver **accurate, context-aware answers**.  
With **voice input, translation, and text-to-speech support**, KrishiGPT ensures accessibility for farmers who prefer speaking over typing.

---

## 🚀 Features  
- 🔍 **Context-aware Q&A** – Answers based on **government-approved agricultural PDFs**.  
- 🎤 **Voice Input Support** – Farmers can speak their queries instead of typing.  
- 🗣️ **Text-to-Speech (TTS)** – Responses are read aloud in Hindi or English.  
- 🌐 **Hindi-English Translation** – Auto-detects language and translates as needed.  
- ⚡ **RAG Pipeline** – Combines FAISS vector search with powerful LLMs (Gemini/Groq fallback).  
- 📚 **PDF-based Knowledge Base** – Uses official crop guides and government schemes.  
- 📜 **Explainable Context** – Shows retrieved document chunks for transparency.  

---

## 🛠️ Tech Stack  

| **Component**        | **Technology Used**             |
|----------------------|---------------------------------|
| **LLM**              | Gemini Pro / Groq-hosted LLM    |
| **Vector Store**     | FAISS                           |
| **Embeddings**       | HuggingFace / OpenAI            |
| **RAG Framework**    | LangChain                       |
| **UI**               | Streamlit                       |
| **TTS**              | gTTS / Coqui                    |
| **Voice Input**      | SpeechRecognition / mic_recorder|
| **Doc Parsing**      | PyPDF2 + LangChain loaders      |

---

## 📁 Project Structure  

```plaintext
KrishiGPT/
│
├── app/
│   ├── main.py              # Streamlit frontend (UI for farmers)
│   └── voice_input.py       # Voice input handler (speech-to-text)
│
├── backend/
│   ├── rag_pipeline.py      # RAG pipeline (FAISS + LLM)
│   ├── tts_response.py      # Text-to-Speech generation
│   ├── translate.py         # Hindi-English translation
│   └── language_utils.py    # Language detection and handling
│
├── data/
│   ├── pdfs/                # Government schemes & crop guides
│   ├── cleaned_docs/        # Text chunks for embeddings
│   └── faiss_index/         # FAISS vector index
│
├── notebooks/
│   ├── pdf_cleaning.py      # Script for cleaning & parsing PDFs
│   ├── test_embeddings.py   # Testing embedding generation
│   └── translate_eval.py    # Translation evaluation experiments
│
├── vectorstore/
│   ├── index.faiss          # FAISS vector database
│   └── index.pkl            # Metadata for embeddings
│
│
├── requirements.txt         # Python dependencies
├── .env                     # API keys and configuration
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

---

## 🧪 Setup Instructions  

### **1. Clone the Repository**  
```bash
git clone https://github.com/hittie93/KrishiGPT.git
cd KrishiGPT
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
pip install -r old_requirements.txt
```

### **3. Prepare `.env` File**  
Create a `.env` file in the root directory:  
```
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_api_key
```

### **4. Build FAISS Vector Index**  
```bash
python backend/build_faiss_index.py
```

### **5. Run KrishiGPT**  
```bash
streamlit run app/main.py
```
Open your browser at **http://localhost:8501** to access KrishiGPT.  

---

## ⚠️ Disclaimer  
- KrishiGPT provides **general agricultural guidance** and is **not a replacement for expert agronomists**.  
- Farmers should **verify the advice** with local agricultural experts or official sources before taking decisions.  
