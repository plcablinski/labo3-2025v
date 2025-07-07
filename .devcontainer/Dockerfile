# Dockerfile para un entorno de desarrollo con PyTorch y dependencias básicas
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Instala herramientas básicas
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    nano \
    htop \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements.txt e instala dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
