FROM nvidia/cuda:11.8-runtime-ubuntu20.04

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    poppler-utils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .