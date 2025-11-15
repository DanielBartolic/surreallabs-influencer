# Qwen-Image Serverless API

Serverless image generation API using Qwen-Image Lightning model, deployed on RunPod Serverless with automated GitHub Actions build.

## Overview

This project converts a ComfyUI Qwen-Image workflow into a serverless API that:
- Uses Qwen-Image with Lightning optimization (8-step generation)
- Deploys on RunPod Serverless with auto-scaling
- Builds automatically via GitHub Actions
- Supports custom LoRA models
- Generates high-quality images in 2-3 seconds

## Architecture

- **Models**: Qwen-Image BF16, Lightning 8-steps, VAE, Text Encoder (~20GB total)
- **Platform**: RunPod Serverless (GPU: RTX 4090 recommended)
- **Build**: Automated via GitHub Actions
- **API**: REST API with RunPod endpoints

## Quick Start

### 1. Set Up GitHub Secrets

Go to your repository **Settings** → **Secrets and variables** → **Actions** and add:

- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password

### 2. Push to GitHub

The GitHub Actions workflow will automatically build and push the Docker image:

```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

### 3. Monitor Build

- Go to **Actions** tab in your GitHub repository
- Watch the "Build and Push Docker Image" workflow
- Build takes ~20-30 minutes (downloads models in the cloud)

### 4. Deploy to RunPod

Once the build completes:

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:
   - **Name**: `qwen-image-api`
   - **Container Image**: `diobrando0/qwen-serverless:latest`
   - **Container Disk**: 50 GB
   - **GPU Type**: NVIDIA GeForce RTX 4090
   - **Min Workers**: 0
   - **Max Workers**: 3
   - **Idle Timeout**: 60 seconds
4. Click **Deploy**

### 5. Get API Credentials

After deployment:
- Copy your **Endpoint ID**
- Copy your **API Key** from RunPod settings
- Save both for API calls

## API Usage

### Generate an Image

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A beautiful sunset over mountains, golden hour, photorealistic",
      "negative_prompt": "blurry, low quality",
      "aspect_ratio": "16:9",
      "steps": 8,
      "cfg": 3.5,
      "seed": -1
    }
  }'
```

### Check Job Status

```bash
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of desired image |
| `negative_prompt` | string | "" | What to avoid in the image |
| `aspect_ratio` | string | "1:1" | Preset ratios: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3 |
| `width` | int | 1328 | Image width (overridden by aspect_ratio) |
| `height` | int | 1328 | Image height (overridden by aspect_ratio) |
| `steps` | int | 8 | Sampling steps (8 recommended for Lightning) |
| `cfg` | float | 3.5 | Guidance scale |
| `seed` | int | -1 | Random seed (-1 for random) |
| `lora_url` | string | null | URL to custom LoRA file (.safetensors) |
| `lora_strength` | float | 0.8 | LoRA influence (0-1) |

## Aspect Ratio Presets

| Ratio | Dimensions | Use Case |
|-------|------------|----------|
| 1:1 | 1328×1328 | Square, social media |
| 16:9 | 1664×928 | Landscape, video |
| 9:16 | 928×1664 | Portrait, mobile |
| 4:3 | 1472×1140 | Standard photo |
| 3:4 | 1140×1472 | Portrait photo |
| 3:2 | 1584×1056 | Classic photo |
| 2:3 | 1056×1584 | Portrait |

## Response Format

```json
{
  "success": true,
  "image": "base64_encoded_image_data",
  "seed": 12345678,
  "width": 1328,
  "height": 1328,
  "steps": 8,
  "cfg": 3.5,
  "prompt": "Your prompt",
  "sampler": "dpmpp_2m_cfg_pp",
  "scheduler": "beta"
}
```

## Cost Estimates

Based on RTX 4090 pricing:

- **Generation time**: 2-3 seconds per image
- **Cost per image**: ~$0.0002-0.0004
- **Monthly (1000 images/day)**: ~$10-15
- **No idle costs**: Scales to zero when not in use

## Project Structure

```
surreallabs-influencer/
├── Dockerfile                      # Container definition
├── requirements.txt                # Python dependencies
├── comfyui_to_serverless.py       # Main handler script
├── .github/
│   └── workflows/
│       └── docker-build.yml       # GitHub Actions automation
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git exclusions
└── README.md                      # This file
```

## Development

### Local Testing (Requires GPU)

```bash
docker build -t qwen-serverless:test .
docker run --gpus all -p 8000:8000 qwen-serverless:test
```

### Manual Build and Push

```bash
docker build -t diobrando0/qwen-serverless:latest .
docker push diobrando0/qwen-serverless:latest
```

## Troubleshooting

### Build Fails
- Check GitHub Actions logs
- Verify model download URLs are accessible
- Ensure sufficient build time (up to 30 minutes)

### Deployment Issues
- Verify Docker image exists on Docker Hub
- Check RunPod GPU availability
- Ensure container disk is at least 50GB

### Slow Generation
- Use RTX 4090 or A40 GPUs
- Keep steps at 8 for Lightning model
- Verify models loaded correctly

## Credits

- **Model**: Qwen-Image by Alibaba Cloud
- **Lightning**: Qwen-Image-Lightning by lightx2v
- **Platform**: RunPod Serverless
- **Framework**: ComfyUI

## License

This project is for educational and commercial use. Model licenses apply.

## Support

For issues or questions:
- Check RunPod documentation
- Review ComfyUI GitHub
- Check model documentation on HuggingFace
