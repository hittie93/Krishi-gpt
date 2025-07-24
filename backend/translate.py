from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

# Supported languages
LANGUAGES = {
    "en": "English",
    "hi": "Hindi"
}

# Model names
MODEL_NAME_EN_HI = "Helsinki-NLP/opus-mt-en-hi"
MODEL_NAME_HI_EN = "Helsinki-NLP/opus-mt-hi-en"

# Cache models
tokenizer_cache = {}
model_cache = {}

def load_model(model_name: str):
    if model_name not in tokenizer_cache:
        tokenizer_cache[model_name] = MarianTokenizer.from_pretrained(model_name)
        model_cache[model_name] = MarianMTModel.from_pretrained(model_name)
    return tokenizer_cache[model_name], model_cache[model_name]

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    if src_lang == tgt_lang:
        return text

    if src_lang == "en" and tgt_lang == "hi":
        model_name = MODEL_NAME_EN_HI
    elif src_lang == "hi" and tgt_lang == "en":
        model_name = MODEL_NAME_HI_EN
    else:
        raise ValueError(f"Unsupported translation direction: {src_lang} → {tgt_lang}")

    tokenizer, model = load_model(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs, max_length=512)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Language detection
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in LANGUAGES else "en"
    except:
        return "en"

# Convert to English
def translate_to_english(text: str) -> str:
    src_lang = detect_language(text)
    print(f"🌐 Detected language: {LANGUAGES.get(src_lang, 'Unknown')}")
    return translate(text, src_lang=src_lang, tgt_lang="en")

# Convert back
def translate_from_english(answer: str, original_input: str) -> str:
    src_lang = detect_language(original_input)
    return translate(answer, src_lang="en", tgt_lang="hi") if src_lang == "hi" else answer

# Example usage
if __name__ == "__main__":
    hindi_input = "टमाटर की फसल को कौन सा उर्वरक देना चाहिए?"
    english_input = "What fertilizer is best for tomato crops?"

    print("\n🔁 Hindi → English:", translate_to_english(hindi_input))
    print("🔁 English → English:", translate_to_english(english_input))

    english_answer = "Tomato crops need balanced NPK fertilizer and organic compost."
    print("\n🈯 English Answer → Hindi:", translate_from_english(english_answer, hindi_input))
