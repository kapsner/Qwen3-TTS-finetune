FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  git \
  libsndfile1 \
  sox libsox-fmt-all \
  ffmpeg \
  wget \
  curl \
  nano && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ENV UV_PROJECT_ENVIRONMENT="/usr/"
RUN pip install --no-cache-dir --break-system-packages uv

# Install core dependencies
RUN uv pip install --system --break-system-packages \
    numpy \
    librosa \
    soundfile \
    tqdm \
    transformers \
    accelerate \
    safetensors \
    datasets \
    evaluate \
    huggingface-hub \
    hf_transfer \
    ffmpeg-python \
    gruut \
    cn2an \
    pypinyin \
    jieba \
    nemo_text_processing

# Install qwen-tts
RUN uv pip install --system --break-system-packages qwen-tts

ENV _GLIBCXX_USE_CXX11_ABI=0
RUN uv pip install --system --break-system-packages torch==2.12.0 --no-build-isolation
RUN uv pip install --system --break-system-packages torchaudio==2.11.0 torchvision==0.27.0 torchcodec==0.14.0 --no-build-isolation
RUN MAX_JOBS=3 uv pip install --system --break-system-packages flash-attn==2.8.3.post1 --no-build-isolation

# Install WhisperX (--no-deps to not break pytorch installation)
RUN uv pip install --no-deps --system --break-system-packages whisperx
RUN uv pip install --system --break-system-packages \
    ctranslate2 \
    faster-whisper \
    av \
    pyannote-audio \
    omegaconf

RUN apt-get update && apt-get install -y \
    libcublas-12-0 && \
    rm -rf /var/lib/apt/lists/*

ADD dataset.py /opt/
ADD prepare_data.py /opt/
ADD setup.sh /opt/
ADD sft_12hz.py /opt/
ADD train.sh /opt/
ADD train_from_audio.py /opt/
ADD start.sh /opt/

RUN chmod +x /opt/*.sh

RUN cd /opt && \
    ./setup.sh --auto

CMD ["/opt/start.sh"]
