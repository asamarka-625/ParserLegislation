FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Устанавливаем временную зону и локаль
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Обновляем и устанавливаем системные зависимости для EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    poppler-utils \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopencv-dev \
    ffmpeg \
    wget \
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

# Оптимизация для PyTorch + CUDA 11.8
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV FORCE_CUDA=1

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

# Создаем директорию для кеша моделей EasyOCR
RUN mkdir -p /root/.EasyOCR/model

# Копируем код
COPY . .

# Оптимизация производительности
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1