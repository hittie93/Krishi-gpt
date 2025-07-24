

# 📁 Step 1: Imports
import os
import pdfplumber
import pandas as pd
from tqdm import tqdm

# 📂 Paths
RAW_PDF_DIR = r"C:\Users\saiha\OneDrive\Documents\Krishi_GPT\data\pdfs"
CLEANED_DOCS_DIR = r"C:\Users\saiha\OneDrive\Documents\Krishi_GPT\data\cleaned_docs"

# ✅ Create cleaned_docs folder if it doesn't exist
os.makedirs(CLEANED_DOCS_DIR, exist_ok=True)

# 🧹 Step 2: Cleaning Function
def clean_text(text):
    # Remove newlines and multiple spaces
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return " ".join(cleaned_lines)

# 📄 Step 3: Process each PDF
def extract_clean_save(pdf_path, file_name):
    all_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += "\n" + text

    cleaned = clean_text(all_text)

    # Save as .txt
    txt_path = os.path.join(CLEANED_DOCS_DIR, file_name.replace(".pdf", ".txt"))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # Save as .csv (one chunk per row ~500 chars)
    chunk_size = 500
    chunks = [cleaned[i:i+chunk_size] for i in range(0, len(cleaned), chunk_size)]
    df = pd.DataFrame({"content": chunks})
    csv_path = os.path.join(CLEANED_DOCS_DIR, file_name.replace(".pdf", ".csv"))
    df.to_csv(csv_path, index=False)

# 🚀 Step 4: Run for all PDFs in /data/
pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith(".pdf")]

for pdf_file in tqdm(pdf_files, desc="📚 Processing PDFs"):
    pdf_path = os.path.join(RAW_PDF_DIR, pdf_file)
    extract_clean_save(pdf_path, pdf_file)

print(f"✅ Done. Cleaned files saved to: {CLEANED_DOCS_DIR}")
