# Dockerfile — CUDA + PyTorch + f5-tts
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git wget ca-certificates python3 python3-venv python3-pip build-essential ffmpeg && \
  rm -rf /var/lib/apt/lists/*

# Ensure pip up-to-date
RUN python3 -m pip install --upgrade pip

# Install PyTorch matching CUDA 12.4
RUN pip install "torch==2.4.0+cu124" "torchaudio==2.4.0+cu124" --extra-index-url https://download.pytorch.org/whl/cu124

# Install f5-tts and helpers
RUN pip install f5-tts huggingface_hub gradio

# Create models dir
RUN mkdir -p /models

# Copy entrypoint and Python downloader
COPY entrypoint.sh /app/entrypoint.sh
COPY download_model.py /app/download_model.py
RUN chmod +x /app/entrypoint.sh /app/download_model.py

# Expose Gradio port
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]




