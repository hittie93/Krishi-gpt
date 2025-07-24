import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Always save to project root's vectorstore directory ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# 🧠 Step 2: Load Embedding Model (you can change model here)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 📂 Step 3: Load CSV chunks
CLEANED_DOCS_DIR = r"C:\Users\saiha\OneDrive\Documents\Krishi_GPT\data\cleaned_docs"
all_chunks = []

for fname in os.listdir(CLEANED_DOCS_DIR):
    if fname.endswith(".csv"):
        df = pd.read_csv(os.path.join(CLEANED_DOCS_DIR, fname))
        all_chunks.extend(df['content'].dropna().tolist())

print(f"✅ Loaded {len(all_chunks)} text chunks from cleaned_docs.")

# 🧪 Step 4: Embed all chunks
embeddings = model.encode(all_chunks, show_progress_bar=True)

# 🧩 Step 5: Build FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors.")

# 🔎 Step 6: Test Similarity Search
query = "What fertilizer to use for cotton crop?"
query_vec = model.encode([query])
top_k = 5

D, I = index.search(query_vec, top_k)

print("\n🔍 Top Results for Query:")
for rank, idx in enumerate(I[0]):
    print(f"{rank+1}. {all_chunks[idx][:200]}...\n")

# 💾 Save FAISS index & metadata for reuse in project root's vectorstore/
faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "index.faiss"))
with open(os.path.join(VECTORSTORE_DIR, "index.pkl"), "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ FAISS index and metadata saved to vectorstore/ as index.faiss and index.pkl")