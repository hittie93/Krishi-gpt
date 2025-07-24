from gtts import gTTS
import os
import tempfile
import platform
from typing import Literal

def play_audio(file_path: str):
    """Cross-platform audio playback."""
    try:
        if platform.system() == "Windows":
            os.system(f'start /min wmplayer "{file_path}"')
        elif platform.system() == "Darwin":  # macOS
            os.system(f"afplay '{file_path}'")
        else:  # Linux
            os.system(f"mpg123 '{file_path}'")
    except Exception as e:
        print(f"⚠️ Audio playback failed: {e}")

def text_to_speech(text: str, lang: Literal["hi", "en"] = "hi", autoplay: bool = True) -> str:
    """
    Convert the given text into speech audio (Hindi/English).

    Args:
        text (str): Text to convert.
        lang (str): Language code - 'hi' for Hindi, 'en' for English.
        autoplay (bool): If True, plays the audio immediately.

    Returns:
        str: Path to the saved audio file.
    """
    try:
        # Generate TTS
        tts = gTTS(text=text, lang=lang)

        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)

        # Optionally autoplay
        if autoplay:
            play_audio(temp_file.name)

        return temp_file.name

    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        return ""

def speak_response(text: str, lang: str = "hi"):
    """Simple wrapper to speak a response."""
    text_to_speech(text, lang=lang, autoplay=True)

# Example usage
if __name__ == "__main__":
    hindi_text = "नमस्ते किसान भाई, आपकी फसल के लिए यह जानकारी उपयोगी है।"
    print("🔈 Playing Hindi TTS...")
    text_to_speech(hindi_text, lang="hi")
