FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for voice + TTS)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
