FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Устанавливаем временную зону и локаль
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Обновляем и устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Копируем зависимости
COPY requirements.txt .

# Обновляем pip и устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .