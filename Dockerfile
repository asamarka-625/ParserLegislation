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

# Устанавливаем правильные переменные окружения для CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH}
ENV CUDA_VISIBLE_DEVICES=0

# Создаем симлинки для cuDNN
RUN ldconfig && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8 /usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8 /usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8 /usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8 /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8 /usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8 /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so

# Проверяем установку CUDA и cuDNN
RUN nvcc --version && \
    ldconfig -p | grep cudnn

# Копируем код
COPY . .