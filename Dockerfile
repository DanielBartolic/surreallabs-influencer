# Qwen-Image Serverless Dockerfile
# Based on RunPod PyTorch base image with CUDA support

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install ComfyUI requirements
WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# Install custom nodes
WORKDIR /app/ComfyUI/custom_nodes
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts && \
    git clone https://github.com/rgthree/rgthree-comfy && \
    git clone https://github.com/Jonseed/ComfyUI-Detail-Daemon && \
    git clone https://github.com/ClownsharkBatwing/RES4LYF && \
    git clone https://github.com/digitaljohn/comfyui-propost && \
    git clone https://github.com/gseth/ControlAltAI-Nodes && \
    git clone https://github.com/ltdrdata/was-node-suite-comfyui && \
    git clone https://github.com/M1kep/ComfyLiterals

# Install custom node requirements
RUN for dir in /app/ComfyUI/custom_nodes/*/; do \
    if [ -f "$dir/requirements.txt" ]; then \
        pip install --no-cache-dir -r "$dir/requirements.txt" || true; \
    fi; \
done

# Create model directories
RUN mkdir -p /models \
    /app/ComfyUI/models/checkpoints \
    /app/ComfyUI/models/vae \
    /app/ComfyUI/models/loras \
    /app/ComfyUI/models/text_encoders

# Download Qwen-Image models (combined in single RUN to reduce layers)
WORKDIR /models
RUN echo "Downloading all Qwen models..." && \
    wget -q --show-progress --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 3 \
    -O qwen_image_bf16.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors" && \
    echo "Base model downloaded" && \
    wget -q --show-progress --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 3 \
    -O Qwen-Image-Lightning-8steps-V2.0.safetensors \
    "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0.safetensors" && \
    echo "Lightning model downloaded" && \
    wget -q --show-progress --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 3 \
    -O qwen_image_vae.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" && \
    echo "VAE downloaded" && \
    wget -q --show-progress --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 3 \
    -O qwen_2.5_vl_7b.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" && \
    echo "All models downloaded successfully" && \
    ls -lh /models/

# Copy handler script
WORKDIR /app
COPY comfyui_to_serverless.py /app/handler.py

# Set environment variables
ENV PYTHONPATH=/app/ComfyUI:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Expose port for local testing (optional)
EXPOSE 8000

# Run handler in RunPod mode
CMD ["python", "-u", "handler.py", "--mode", "runpod"]
