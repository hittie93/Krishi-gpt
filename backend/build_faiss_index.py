import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Step 1: Load your documents
docs_folder = "data/cleaned_docs"
documents = []

for file in os.listdir(docs_folder):
    if file.endswith(".txt"):
        with open(os.path.join(docs_folder, file), "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(Document(page_content=content))

print(f"📄 Loaded {len(documents)} documents for indexing.")

# Step 2: Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Build FAISS index
vectorstore = FAISS.from_documents(documents, embedding_model)

# Step 4: Save FAISS index
os.makedirs("data/faiss_index", exist_ok=True)
vectorstore.save_local("data/faiss_index")

print("✅ FAISS index built and saved at data/faiss_index")
