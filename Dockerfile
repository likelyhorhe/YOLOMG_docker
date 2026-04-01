FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/likelyhorhe/YOLOMG_docker.git .

RUN pip install --no-cache-dir -r requirements.txt

RUN wget -q https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O yolov5s.pt

EXPOSE 6006
