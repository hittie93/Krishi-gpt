from langdetect import detect, DetectorFactory

# Ensures consistent results from langdetect
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {
    "en": "english",
    "hi": "hindi"
}

def detect_language(text: str) -> str:
    """
    Detects the language of the input text.
    Returns 'english' or 'hindi' for known languages, 
    otherwise returns the ISO language code.
    """
    try:
        lang_code = detect(text)
        return SUPPORTED_LANGUAGES.get(lang_code, lang_code)
    except Exception:
        return "unknown"
