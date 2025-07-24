from transformers import pipeline
import pandas as pd
import os
import random

# --- Configuration ---
CLEANED_DIR = r"C:\Users\saiha\OneDrive\Documents\Krishi_GPT\data\cleaned_docs"
NUM_SAMPLES = 5  # Number of samples to test

# --- Load cleaned text chunks ---
all_texts = []

for file in os.listdir(CLEANED_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(CLEANED_DIR, file))
        if 'text' in df.columns:
            all_texts.extend(df['text'].dropna().tolist())
        elif 'content' in df.columns:
            all_texts.extend(df['content'].dropna().tolist())
        else:
            print(f"⚠️  Skipping {file}: no 'text' or 'content' column found. Columns are: {list(df.columns)}")

if not all_texts:
    raise ValueError("No text or content data found in any CSV files in the cleaned_docs directory.")

print(f"✅ Loaded {len(all_texts)} cleaned chunks.")

# --- Random sampling ---
sampled_texts = random.sample(all_texts, min(NUM_SAMPLES, len(all_texts)))

# --- Load translation pipelines ---
translator_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
translator_te = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")

# --- Translate and show ---
for i, text in enumerate(sampled_texts):
    print(f"\n🧾 Original Text {i+1}:\n{text}\n")

    hi = translator_hi(text, max_length=512)[0]['translation_text']
    print(f"🔁 Hindi:\n{hi}\n")

    te = translator_te(text, max_length=512)[0]['translation_text']
    print(f"🌾 Telugu:\n{te}\n")