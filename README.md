# 🌾 KrishiGPT – किसान का AI साथी  

![KrishiGPT Banner](assets/krishigpt_banner.png)  

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)  
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)  

---

## **🚀 About KrishiGPT**

**KrishiGPT** is an AI-powered virtual assistant designed to help **Indian farmers** by providing **expert agricultural advice** in both **Hindi and English**.  
It uses **Generative AI + RAG (Retrieval-Augmented Generation)** to answer queries about:  

- Crop management  
- Fertilizers & pesticides  
- Weather information  
- Government schemes  
- Best farming practices  

**Key Highlights:**  
- 🎤 **Voice Input (Hindi/English)**  
- 🔊 **Text-to-Speech Answer**  
- 🔍 **RAG (FAISS Vector Search) for Context Retrieval**  
- 🤖 **Powered by Gemini/Groq LLM**  
- 🌐 **Bilingual Support (Hindi & English)**  

---

## **🖼️ Demo**

> Add a screenshot or gif demo here:  
![KrishiGPT Screenshot](assets/krishigpt_ui.png)

---

## **✨ Features**
- **Voice & Text Input** – Ask questions in Hindi or English.  
- **Explainable Responses** – Retrieves relevant chunks from agricultural PDFs.  
- **TTS Support** – Listen to AI's response.  
- **Modern UI** – **Glassmorphism + Animated Gradient** using **Streamlit**.  
- **Offline RAG** – Uses FAISS to index PDF knowledge base.  

---

## **📂 Project Structure**
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
│   ├── pdfs/                # Government schemes & crop guides (raw PDFs)
│   ├── cleaned_docs/        # Cleaned and chunked documents
│   └── faiss_index/         # FAISS vector index for semantic search
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
└── 


