FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# Обновляем список пакетов ПЕРВЫМ делом
RUN apt-get update

# Устанавливаем локаль (упрощенная версия)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
RUN apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Устанавливаем системные зависимости
RUN apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Настраиваем локаль для Python (без пакета locales)
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python пакеты
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .