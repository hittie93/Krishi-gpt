# backend/voice_input.py
import io
from typing import Optional

import speech_recognition as sr
from pydub import AudioSegment


def _convert_to_wav(audio_bytes: bytes) -> bytes:
    """
    Convert arbitrary audio bytes (webm/mp3/m4a/etc.) to PCM WAV using pydub.
    Requires ffmpeg/avconv installed and available on PATH.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")  # default is PCM WAV
    return wav_io.getvalue()


def transcribe_audio_bytes(
    audio_bytes: bytes,
    language_hint: str = "hi-IN",
    fallback_langs: Optional[list[str]] = None,
) -> str:
    """
    Transcribe raw audio bytes to text using Google Web Speech API.
    1) Converts to WAV
    2) Tries Hindi first, then falls back to given languages.
    """
    if fallback_langs is None:
        fallback_langs = ["en-IN", "en-US", "hi-IN"]

    recognizer = sr.Recognizer()

    try:
        wav_bytes = _convert_to_wav(audio_bytes)
    except Exception as e:
        # If conversion fails, try to read as-is (in case it was already WAV/PCM)
        wav_bytes = audio_bytes

    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)

        # Try primary hint first, then fallbacks
        for lang in [language_hint] + fallback_langs:
            try:
                text = recognizer.recognize_google(audio_data, language=lang)
                if text.strip():
                    return text
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                # API/network error
                raise RuntimeError(f"Speech API error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Could not process audio: {e}") from e

    return ""


def listen_to_voice_cli() -> str:
    """
    Simple CLI microphone capture (not used in Streamlit, but handy for quick tests).
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="hi-IN")
        print(f"🗣 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"❌ Error with speech recognition service: {e}")
        return ""
