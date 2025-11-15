#!/bin/bash
# Build script to run on RunPod instance
# This script builds the Docker image and pushes to Docker Hub

set -e  # Exit on error

echo "======================================"
echo "RunPod Docker Build Script"
echo "======================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Configuration
DOCKER_USERNAME="diobrando0"
DOCKER_IMAGE="qwen-serverless"
DOCKER_TAG="latest"

echo "Docker Hub Username: $DOCKER_USERNAME"
echo "Image Name: $DOCKER_IMAGE:$DOCKER_TAG"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    echo "Docker installed successfully!"
else
    echo "Docker is already installed."
fi

echo ""
echo "======================================"
echo "Step 1: Docker Hub Login"
echo "======================================"
echo "Please enter your Docker Hub credentials:"
docker login

echo ""
echo "======================================"
echo "Step 2: Clone Repository"
echo "======================================"
if [ -d "surreallabs-influencer" ]; then
    echo "Repository already exists. Pulling latest changes..."
    cd surreallabs-influencer
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/DanielBartolic/surreallabs-influencer.git
    cd surreallabs-influencer
fi

echo ""
echo "======================================"
echo "Step 3: Building Docker Image"
echo "======================================"
echo "This will take 20-30 minutes..."
echo "Starting build at: $(date)"
echo ""

docker build -t ${DOCKER_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG} .

echo ""
echo "Build completed at: $(date)"
echo ""

echo "======================================"
echo "Step 4: Pushing to Docker Hub"
echo "======================================"
docker push ${DOCKER_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG}

echo ""
echo "======================================"
echo "✅ SUCCESS!"
echo "======================================"
echo ""
echo "Your Docker image is now available at:"
echo "  ${DOCKER_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG}"
echo ""
echo "Next steps:"
echo "1. Go to https://www.runpod.io/console/serverless"
echo "2. Create a new endpoint"
echo "3. Use image: ${DOCKER_USERNAME}/${DOCKER_IMAGE}:${DOCKER_TAG}"
echo "4. Select GPU: RTX 4090"
echo "5. Container Disk: 50 GB"
echo ""
echo "You can now terminate this RunPod instance!"
echo "======================================"
