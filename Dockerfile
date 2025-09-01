# StyleTTS2 API - Dockerfile (NGC PyTorch base)
# Build args allow selecting the exact NGC tag (e.g., 24.06-py3, 24.08-py3)
ARG NGC_TAG=24.06-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

# System deps: espeak-ng (phonemizer backend), ffmpeg (pydub), libsndfile1 (soundfile), build-essential (cpp ext), git
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng ffmpeg libsndfile1 build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/StyleTTS2

# Python deps
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Project files
COPY . .

# Entrypoint
RUN chmod +x docker-entrypoint.sh || true

# Sensible defaults (can be overridden at runtime)
ENV PROMETHEUS=1 \
    ENABLE_AMP=1 \
    TORCH_NUM_THREADS=1 \
    MODEL_AUTO_DOWNLOAD=1 \
    MODELS="LJSpeech,LibriTTS"

EXPOSE 8000
ENTRYPOINT ["bash", "./docker-entrypoint.sh"]