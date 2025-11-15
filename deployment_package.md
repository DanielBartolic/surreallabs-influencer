# Deployment Package for Qwen-Image Serverless

## Quick Start Files

### 1. Dockerfile
```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl unzip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Install Python packages
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir \
    transformers==4.36.0 \
    diffusers==0.24.0 \
    safetensors==0.4.1 \
    accelerate==0.25.0 \
    xformers==0.0.22 \
    einops \
    omegaconf \
    pillow \
    numpy \
    runpod \
    requests \
    aiohttp

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
RUN mkdir -p /models /app/ComfyUI/models/checkpoints /app/ComfyUI/models/vae /app/ComfyUI/models/loras

# Download models (with retry logic)
WORKDIR /models

# Download base model (large file, may need multiple attempts)
RUN curl -L --retry 3 --retry-delay 5 -o qwen_image_bf16.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors"

# Download Lightning model
RUN curl -L --retry 3 --retry-delay 5 -o Qwen-Image-Lightning-8steps-V2.0.safetensors \
    "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0.safetensors"

# Download VAE
RUN curl -L --retry 3 --retry-delay 5 -o qwen_image_vae.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

# Download text encoder
RUN curl -L --retry 3 --retry-delay 5 -o qwen_2.5_vl_7b.safetensors \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors"

# Copy handler script
WORKDIR /app
COPY comfyui_to_serverless.py .
COPY qwen_influencer_generator.json workflow.json

# Set environment variables
ENV PYTHONPATH=/app/ComfyUI:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run handler
CMD ["python", "-u", "comfyui_to_serverless.py", "--mode", "runpod"]
```

### 2. docker-compose.yml (for local testing)
```yaml
version: '3.8'

services:
  qwen-serverless:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - RUNPOD_DEBUG_LEVEL=INFO
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
      - ./outputs:/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 3. runpod_deploy.sh
```bash
#!/bin/bash

# RunPod Deployment Script
set -e

# Configuration
DOCKER_REPO="your-dockerhub-username/qwen-serverless"
VERSION="latest"
RUNPOD_API_KEY="your-runpod-api-key"
ENDPOINT_NAME="qwen-image-api"

echo "Building Docker image..."
docker build -t $DOCKER_REPO:$VERSION .

echo "Pushing to Docker Hub..."
docker push $DOCKER_REPO:$VERSION

echo "Creating RunPod endpoint..."
curl -X POST https://api.runpod.ai/v2/endpoints \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "'$ENDPOINT_NAME'",
    "templateId": "runpod-pytorch",
    "imageName": "'$DOCKER_REPO':'$VERSION'",
    "gpuType": "NVIDIA GeForce RTX 4090",
    "minWorkers": 0,
    "maxWorkers": 3,
    "idleTimeout": 60,
    "scalerType": "QUEUE_DEPTH",
    "scalerValue": 1,
    "containerDiskInGb": 50,
    "volumeInGb": 50
  }'

echo "Deployment complete!"
```

### 4. Website Integration - index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen-Image Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .main {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        .controls {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        label {
            font-weight: 600;
            color: #333;
            font-size: 14px;
        }
        
        input, textarea, select {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="range"] {
            flex: 1;
        }
        
        .slider-value {
            min-width: 50px;
            text-align: right;
            font-weight: 600;
            color: #667eea;
        }
        
        .btn {
            padding: 14px 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .preview {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .image-container {
            aspect-ratio: 1;
            background: #f5f5f5;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        
        .placeholder {
            color: #999;
            font-size: 18px;
        }
        
        .loading {
            position: absolute;
            inset: 0;
            background: rgba(255,255,255,0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 10px;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .metadata {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            color: #666;
        }
        
        .error {
            background: #fee;
            color: #c00;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        
        .advanced-toggle {
            cursor: pointer;
            user-select: none;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
            color: #667eea;
        }
        
        .advanced-settings {
            display: none;
            gap: 20px;
            flex-direction: column;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-top: 10px;
        }
        
        .advanced-settings.open {
            display: flex;
        }
        
        @media (max-width: 768px) {
            .main {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Qwen-Image Generator</h1>
            <p>Powered by Qwen-Image Lightning (8-step generation)</p>
        </div>
        
        <div class="main">
            <div class="controls">
                <div class="input-group">
                    <label for="prompt">Prompt</label>
                    <textarea id="prompt" placeholder="Describe what you want to generate...">A beautiful sunset over mountains, golden hour lighting, photorealistic</textarea>
                </div>
                
                <div class="input-group">
                    <label for="negative">Negative Prompt</label>
                    <textarea id="negative" placeholder="What to avoid...">blurry, low quality, distorted</textarea>
                </div>
                
                <div class="input-group">
                    <label for="aspect">Aspect Ratio</label>
                    <select id="aspect">
                        <option value="1:1">1:1 Square (1328×1328)</option>
                        <option value="16:9">16:9 Landscape (1664×928)</option>
                        <option value="9:16">9:16 Portrait (928×1664)</option>
                        <option value="4:3">4:3 Standard (1472×1140)</option>
                        <option value="3:4">3:4 Portrait (1140×1472)</option>
                        <option value="3:2">3:2 Photo (1584×1056)</option>
                        <option value="2:3">2:3 Portrait (1056×1584)</option>
                    </select>
                </div>
                
                <div class="advanced-toggle" onclick="toggleAdvanced()">
                    ⚙️ Advanced Settings ▼
                </div>
                
                <div class="advanced-settings" id="advanced">
                    <div class="input-group">
                        <label for="steps">Steps: <span id="steps-value">8</span></label>
                        <div class="slider-container">
                            <input type="range" id="steps" min="4" max="20" value="8">
                            <span class="slider-value">8</span>
                        </div>
                    </div>
                    
                    <div class="input-group">
                        <label for="cfg">CFG Scale: <span id="cfg-value">3.5</span></label>
                        <div class="slider-container">
                            <input type="range" id="cfg" min="1" max="10" step="0.5" value="3.5">
                            <span class="slider-value">3.5</span>
                        </div>
                    </div>
                    
                    <div class="input-group">
                        <label for="seed">Seed (-1 for random)</label>
                        <input type="number" id="seed" value="-1" min="-1">
                    </div>
                    
                    <div class="input-group">
                        <label for="lora">Custom LoRA (optional)</label>
                        <input type="file" id="lora" accept=".safetensors">
                    </div>
                    
                    <div class="input-group">
                        <label for="lora-strength">LoRA Strength: <span id="lora-strength-value">0.8</span></label>
                        <div class="slider-container">
                            <input type="range" id="lora-strength" min="0" max="1" step="0.1" value="0.8">
                            <span class="slider-value">0.8</span>
                        </div>
                    </div>
                </div>
                
                <button class="btn" id="generate-btn" onclick="generateImage()">
                    Generate Image
                </button>
                
                <div id="error-container"></div>
            </div>
            
            <div class="preview">
                <div class="image-container" id="image-container">
                    <div class="placeholder">Your image will appear here</div>
                </div>
                
                <div class="metadata" id="metadata" style="display: none;">
                    <strong>Generation Details:</strong>
                    <div id="metadata-content"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Configuration
        const API_ENDPOINT = 'YOUR_RUNPOD_ENDPOINT_URL'; // Replace with your RunPod endpoint
        const API_KEY = 'YOUR_RUNPOD_API_KEY'; // Replace with your API key
        
        // State
        let isGenerating = false;
        
        // UI Functions
        function toggleAdvanced() {
            const advanced = document.getElementById('advanced');
            advanced.classList.toggle('open');
        }
        
        // Update slider values
        document.getElementById('steps').addEventListener('input', (e) => {
            document.getElementById('steps-value').textContent = e.target.value;
            e.target.nextElementSibling.textContent = e.target.value;
        });
        
        document.getElementById('cfg').addEventListener('input', (e) => {
            document.getElementById('cfg-value').textContent = e.target.value;
            e.target.nextElementSibling.textContent = e.target.value;
        });
        
        document.getElementById('lora-strength').addEventListener('input', (e) => {
            document.getElementById('lora-strength-value').textContent = e.target.value;
            e.target.nextElementSibling.textContent = e.target.value;
        });
        
        // Upload LoRA to temporary storage
        async function uploadLora(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            // Upload to your file hosting service
            // This is a placeholder - implement your own file upload
            const response = await fetch('/api/upload-lora', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            return data.url;
        }
        
        // Generate Image
        async function generateImage() {
            if (isGenerating) return;
            
            isGenerating = true;
            const btn = document.getElementById('generate-btn');
            const container = document.getElementById('image-container');
            const errorContainer = document.getElementById('error-container');
            
            btn.disabled = true;
            btn.textContent = 'Generating...';
            errorContainer.innerHTML = '';
            
            // Show loading
            container.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <div>Generating your image...</div>
                </div>
            `;
            
            try {
                // Get parameters
                const params = {
                    prompt: document.getElementById('prompt').value,
                    negative_prompt: document.getElementById('negative').value,
                    aspect_ratio: document.getElementById('aspect').value,
                    steps: parseInt(document.getElementById('steps').value),
                    cfg: parseFloat(document.getElementById('cfg').value),
                    seed: parseInt(document.getElementById('seed').value)
                };
                
                // Handle LoRA upload
                const loraFile = document.getElementById('lora').files[0];
                if (loraFile) {
                    params.lora_url = await uploadLora(loraFile);
                    params.lora_strength = parseFloat(document.getElementById('lora-strength').value);
                }
                
                // Call API
                const response = await fetch(`${API_ENDPOINT}/run`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${API_KEY}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ input: params })
                });
                
                const result = await response.json();
                
                // Handle async job
                let jobResult = result;
                if (result.status === 'IN_QUEUE' || result.status === 'IN_PROGRESS') {
                    jobResult = await pollForResult(result.id);
                }
                
                // Display image
                if (jobResult.success !== false) {
                    container.innerHTML = `<img src="data:image/png;base64,${jobResult.image}" alt="Generated image">`;
                    
                    // Show metadata
                    const metadata = document.getElementById('metadata');
                    const metadataContent = document.getElementById('metadata-content');
                    metadata.style.display = 'block';
                    metadataContent.innerHTML = `
                        Seed: ${jobResult.seed}<br>
                        Size: ${jobResult.width}×${jobResult.height}<br>
                        Steps: ${jobResult.steps}<br>
                        CFG: ${jobResult.cfg}<br>
                        Sampler: ${jobResult.sampler || 'dpmpp_2m_cfg_pp'}<br>
                        Scheduler: ${jobResult.scheduler || 'beta'}
                    `;
                    
                    // Add download button
                    const downloadBtn = document.createElement('button');
                    downloadBtn.className = 'btn';
                    downloadBtn.style.marginTop = '10px';
                    downloadBtn.textContent = '⬇ Download Image';
                    downloadBtn.onclick = () => downloadImage(jobResult.image, `qwen_${jobResult.seed}.png`);
                    metadata.appendChild(downloadBtn);
                } else {
                    throw new Error(jobResult.error || 'Generation failed');
                }
                
            } catch (error) {
                console.error('Generation error:', error);
                container.innerHTML = '<div class="placeholder">Generation failed</div>';
                errorContainer.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            } finally {
                isGenerating = false;
                btn.disabled = false;
                btn.textContent = 'Generate Image';
            }
        }
        
        // Poll for async result
        async function pollForResult(jobId, maxAttempts = 60) {
            for (let i = 0; i < maxAttempts; i++) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                const response = await fetch(`${API_ENDPOINT}/status/${jobId}`, {
                    headers: {
                        'Authorization': `Bearer ${API_KEY}`
                    }
                });
                
                const result = await response.json();
                
                if (result.status === 'COMPLETED') {
                    return result.output;
                } else if (result.status === 'FAILED') {
                    throw new Error(result.error || 'Job failed');
                }
                
                // Update loading text
                const loading = document.querySelector('.loading div:last-child');
                if (loading) {
                    loading.textContent = `Generating... (${i + 1}s)`;
                }
            }
            
            throw new Error('Generation timeout');
        }
        
        // Download image
        function downloadImage(base64, filename) {
            const link = document.createElement('a');
            link.href = `data:image/png;base64,${base64}`;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
        
        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                generateImage();
            }
        });
    </script>
</body>
</html>
```

### 5. API Backend (Node.js) - server.js
```javascript
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const multer = require('multer');
const AWS = require('aws-sdk');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// Configure AWS S3 for LoRA storage
const s3 = new AWS.S3({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configuration
const RUNPOD_ENDPOINT = process.env.RUNPOD_ENDPOINT;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;

// Rate limiting
const rateLimit = new Map();

function checkRateLimit(ip) {
    const now = Date.now();
    const limit = 10; // 10 requests per minute
    const window = 60000; // 1 minute
    
    if (!rateLimit.has(ip)) {
        rateLimit.set(ip, []);
    }
    
    const requests = rateLimit.get(ip).filter(time => now - time < window);
    
    if (requests.length >= limit) {
        return false;
    }
    
    requests.push(now);
    rateLimit.set(ip, requests);
    return true;
}

// Upload LoRA endpoint
app.post('/api/upload-lora', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }
        
        const fileKey = `loras/${uuidv4()}.safetensors`;
        
        const params = {
            Bucket: process.env.S3_BUCKET,
            Key: fileKey,
            Body: req.file.buffer,
            ContentType: 'application/octet-stream'
        };
        
        await s3.upload(params).promise();
        
        const url = `https://${process.env.S3_BUCKET}.s3.${process.env.AWS_REGION}.amazonaws.com/${fileKey}`;
        
        res.json({ url });
    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ error: 'Upload failed' });
    }
});

// Generate image endpoint
app.post('/api/generate', async (req, res) => {
    // Check rate limit
    const clientIp = req.ip || req.connection.remoteAddress;
    if (!checkRateLimit(clientIp)) {
        return res.status(429).json({ error: 'Rate limit exceeded. Please wait a minute.' });
    }
    
    try {
        // Validate input
        const { prompt, negative_prompt, aspect_ratio, steps, cfg, seed, lora_url, lora_strength } = req.body;
        
        if (!prompt || prompt.length > 1000) {
            return res.status(400).json({ error: 'Invalid prompt' });
        }
        
        // Call RunPod API
        const response = await axios.post(
            `${RUNPOD_ENDPOINT}/run`,
            {
                input: {
                    prompt,
                    negative_prompt: negative_prompt || '',
                    aspect_ratio: aspect_ratio || '1:1',
                    steps: steps || 8,
                    cfg: cfg || 3.5,
                    seed: seed || -1,
                    lora_url,
                    lora_strength: lora_strength || 0.8
                }
            },
            {
                headers: {
                    'Authorization': `Bearer ${RUNPOD_API_KEY}`,
                    'Content-Type': 'application/json'
                }
            }
        );
        
        res.json(response.data);
    } catch (error) {
        console.error('Generation error:', error);
        res.status(500).json({ error: 'Generation failed' });
    }
});

// Check job status
app.get('/api/status/:jobId', async (req, res) => {
    try {
        const response = await axios.get(
            `${RUNPOD_ENDPOINT}/status/${req.params.jobId}`,
            {
                headers: {
                    'Authorization': `Bearer ${RUNPOD_API_KEY}`
                }
            }
        );
        
        res.json(response.data);
    } catch (error) {
        console.error('Status check error:', error);
        res.status(500).json({ error: 'Status check failed' });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'healthy' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
```

### 6. Environment Variables (.env)
```bash
# RunPod Configuration
RUNPOD_ENDPOINT=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID
RUNPOD_API_KEY=your_runpod_api_key

# AWS S3 Configuration (for LoRA storage)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name

# Server Configuration
PORT=3000
NODE_ENV=production
```

### 7. Package.json
```json
{
  "name": "qwen-image-api",
  "version": "1.0.0",
  "description": "API server for Qwen-Image serverless generation",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "axios": "^1.6.0",
    "multer": "^1.4.5-lts.1",
    "aws-sdk": "^2.1450.0",
    "uuid": "^9.0.0",
    "dotenv": "^16.3.1",
    "helmet": "^7.0.0",
    "compression": "^1.7.4"
  },
  "devDependencies": {
    "nodemon": "^3.0.0",
    "jest": "^29.0.0"
  }
}
```

## Testing Locally

1. **Build and test the Docker image:**
```bash
# Build
docker build -t qwen-serverless:test .

# Run locally
docker run --gpus all -p 8000:8000 qwen-serverless:test
```

2. **Test the API:**
```bash
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A beautiful sunset over mountains",
      "width": 1328,
      "height": 1328,
      "steps": 8
    }
  }'
```

## Deployment to RunPod

1. **Push to Docker Hub:**
```bash
docker tag qwen-serverless:test yourusername/qwen-serverless:latest
docker push yourusername/qwen-serverless:latest
```

2. **Create RunPod Endpoint:**
- Go to [RunPod Console](https://www.runpod.io/console/serverless)
- Click "New Endpoint"
- Configure:
  - Name: `qwen-image-api`
  - Docker Image: `yourusername/qwen-serverless:latest`
  - GPU: RTX 4090 (recommended for cost/performance)
  - Min Workers: 0
  - Max Workers: 3
  - Idle Timeout: 60 seconds
  - Container Disk: 50GB
  - Volume: 50GB (for model caching)

3. **Get your endpoint URL and API key from RunPod**

4. **Update your website configuration with the endpoint details**

## Cost Optimization

- **RTX 4090**: ~$0.44/hour when active, $0 when idle
- **Average generation**: 2-3 seconds
- **Cost per image**: ~$0.0002-0.0004
- **Monthly cost (1000 images/day)**: ~$10-15

## Monitoring

Use RunPod's built-in monitoring or integrate with:
- **Grafana** for metrics visualization
- **Sentry** for error tracking
- **CloudWatch** for AWS integration

## Support

For issues or questions:
1. Check RunPod documentation
2. Review ComfyUI GitHub issues
3. Check model documentation on HuggingFace
