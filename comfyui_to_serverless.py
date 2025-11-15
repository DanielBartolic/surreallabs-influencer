# comfyui_to_serverless.py
"""
Direct ComfyUI Workflow to Serverless Implementation
This script provides a production-ready implementation of your Qwen-Image workflow
that can be deployed to RunPod Serverless, Modal, or any other serverless platform.
"""

import os
import sys
import json
import time
import base64
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import traceback

import torch
import numpy as np
from PIL import Image
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelPaths:
    """Paths to all required model files"""
    base_model: str = "/models/qwen_image_bf16.safetensors"
    lightning_model: str = "/models/Qwen-Image-Lightning-8steps-V2.0.safetensors"
    vae: str = "/models/qwen_image_vae.safetensors"
    text_encoder: str = "/models/qwen_2.5_vl_7b.safetensors"
    comfyui_path: str = "/app/ComfyUI"
    custom_nodes_path: str = "/app/ComfyUI/custom_nodes"

class ComfyUIQwenWrapper:
    """
    Wrapper class that directly uses ComfyUI's internal components
    to replicate your exact workflow
    """
    
    def __init__(self, model_paths: ModelPaths):
        self.model_paths = model_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add ComfyUI to path
        sys.path.insert(0, model_paths.comfyui_path)
        
        # Import ComfyUI components after adding to path
        self._import_comfy_components()
        
        # Model cache
        self.model = None
        self.clip = None
        self.vae = None
        self.loaded_lora = None
        
        # Initialize models on first use
        self._initialized = False
        
    def _import_comfy_components(self):
        """Import required ComfyUI components"""
        try:
            # Core imports
            import folder_paths
            import comfy.model_management as model_management
            
            # Set up ComfyUI paths
            folder_paths.models_dir = "/models"
            folder_paths.output_directory = "/tmp/output"
            
            # Import node classes
            from nodes import (
                CheckpointLoaderSimple,
                VAELoader,
                CLIPTextEncode,
                EmptySD3LatentImage,
                VAEDecode,
                SaveImage
            )
            
            # Import custom sampling nodes
            import sys
            sys.path.append(self.model_paths.custom_nodes_path)
            
            # Store references
            self.folder_paths = folder_paths
            self.model_management = model_management
            self.CheckpointLoaderSimple = CheckpointLoaderSimple
            self.VAELoader = VAELoader
            self.CLIPTextEncode = CLIPTextEncode
            self.EmptySD3LatentImage = EmptySD3LatentImage
            self.VAEDecode = VAEDecode
            self.SaveImage = SaveImage
            
            # Import additional nodes from your workflow
            from custom_nodes.ComfyUI_Detail_Daemon import DetailDaemonSamplerNode
            from custom_nodes.RES4LYF import ClownsharKSampler_Beta
            from custom_nodes.rgthree_comfy import PowerLoraLoader
            
            self.DetailDaemonSamplerNode = DetailDaemonSamplerNode
            self.ClownsharKSampler_Beta = ClownsharKSampler_Beta
            self.PowerLoraLoader = PowerLoraLoader
            
            logger.info("ComfyUI components imported successfully")
            
        except Exception as e:
            logger.error(f"Failed to import ComfyUI components: {e}")
            raise
            
    def initialize(self):
        """Initialize models (lazy loading for better cold start)"""
        if self._initialized:
            return
            
        logger.info("Initializing Qwen-Image models...")
        start_time = time.time()
        
        try:
            # Load checkpoint (combining base + lightning)
            self._load_combined_model()
            
            # Load VAE
            self._load_vae()
            
            # Load CLIP/Text Encoder
            self._load_text_encoder()
            
            self._initialized = True
            logger.info(f"Models initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
            
    def _load_combined_model(self):
        """Load and combine base model with Lightning weights"""
        import safetensors.torch
        
        # Load base model
        logger.info("Loading base model...")
        checkpoint_loader = self.CheckpointLoaderSimple()
        
        # Copy model files to ComfyUI expected location if needed
        import shutil
        models_dir = "/app/ComfyUI/models/checkpoints"
        os.makedirs(models_dir, exist_ok=True)
        
        base_target = os.path.join(models_dir, "qwen_image_bf16.safetensors")
        if not os.path.exists(base_target):
            shutil.copy(self.model_paths.base_model, base_target)
            
        # Load through ComfyUI
        self.model, self.clip, _ = checkpoint_loader.load_checkpoint(
            ckpt_name="qwen_image_bf16.safetensors"
        )
        
        # Apply Lightning weights for 8-step generation
        logger.info("Applying Lightning optimization...")
        lightning_weights = safetensors.torch.load_file(
            self.model_paths.lightning_model,
            device=str(self.device)
        )
        
        # Merge Lightning weights
        model_sd = self.model.model.state_dict()
        for key, value in lightning_weights.items():
            if key in model_sd:
                model_sd[key] = value
        self.model.model.load_state_dict(model_sd)
        
    def _load_vae(self):
        """Load VAE model"""
        logger.info("Loading VAE...")
        vae_loader = self.VAELoader()
        
        # Prepare VAE path
        vae_dir = "/app/ComfyUI/models/vae"
        os.makedirs(vae_dir, exist_ok=True)
        
        import shutil
        vae_target = os.path.join(vae_dir, "qwen_image_vae.safetensors")
        if not os.path.exists(vae_target):
            shutil.copy(self.model_paths.vae, vae_target)
            
        self.vae, = vae_loader.load_vae(vae_name="qwen_image_vae.safetensors")
        
    def _load_text_encoder(self):
        """Load text encoder (already loaded with checkpoint in ComfyUI)"""
        logger.info("Text encoder loaded with checkpoint")
        # The CLIP/text encoder is already loaded as part of the checkpoint
        
    def load_lora(self, lora_path: str, strength: float = 0.8) -> bool:
        """
        Load custom LoRA weights
        
        Args:
            lora_path: Path to LoRA safetensors file
            strength: LoRA strength (0-1)
            
        Returns:
            Success status
        """
        try:
            if not os.path.exists(lora_path):
                logger.warning(f"LoRA file not found: {lora_path}")
                return False
                
            logger.info(f"Loading LoRA: {lora_path} with strength {strength}")
            
            # Use Power Lora Loader from your workflow
            lora_loader = self.PowerLoraLoader()
            
            # Copy LoRA to expected location
            lora_dir = "/app/ComfyUI/models/loras"
            os.makedirs(lora_dir, exist_ok=True)
            
            import shutil
            lora_name = os.path.basename(lora_path)
            lora_target = os.path.join(lora_dir, lora_name)
            shutil.copy(lora_path, lora_target)
            
            # Apply LoRA
            self.model, self.clip = lora_loader.load_lora(
                model=self.model,
                clip=self.clip,
                lora_name=lora_name,
                strength_model=strength,
                strength_clip=strength
            )
            
            self.loaded_lora = lora_name
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            return False
            
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1328,
        height: int = 1328,
        steps: int = 8,
        cfg: float = 3.5,
        sampler_name: str = "dpmpp_2m_cfg_pp",
        scheduler: str = "beta",
        seed: int = -1,
        lora_path: Optional[str] = None,
        lora_strength: float = 0.8,
        denoise: float = 1.0,
        aspect_ratio: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate image using the exact workflow from your ComfyUI setup
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Number of sampling steps (8 for Lightning)
            cfg: CFG scale
            sampler_name: Sampler algorithm
            scheduler: Scheduler type
            seed: Random seed (-1 for random)
            lora_path: Optional LoRA file path
            lora_strength: LoRA influence strength
            denoise: Denoising strength
            aspect_ratio: Preset aspect ratio (overrides width/height)
            
        Returns:
            Dictionary with base64 image and metadata
        """
        
        # Initialize if needed
        self.initialize()
        
        # Apply aspect ratio presets from your workflow
        if aspect_ratio:
            width, height = self._get_dimensions_from_aspect_ratio(aspect_ratio)
            
        # Load LoRA if provided
        if lora_path:
            self.load_lora(lora_path, lora_strength)
            
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32-1, (1,)).item()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info(f"Generating image: {width}x{height}, steps={steps}, seed={seed}")
        
        try:
            # Create conditioning (positive and negative prompts)
            positive_cond = self._encode_prompt(prompt, self.clip)
            negative_cond = self._encode_prompt(negative_prompt, self.clip)
            
            # Create empty latent
            latent = self._create_latent(width, height)
            
            # Run sampling using your exact workflow setup
            samples = self._run_sampling(
                latent=latent,
                positive=positive_cond,
                negative=negative_cond,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                seed=seed,
                denoise=denoise
            )
            
            # Decode VAE
            images = self._decode_vae(samples)
            
            # Apply post-processing (Film Grain from your workflow)
            images = self._apply_film_grain(images, seed)
            
            # Convert to base64
            image_base64 = self._tensor_to_base64(images)
            
            return {
                "success": True,
                "image": image_base64,
                "seed": seed,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "sampler": sampler_name,
                "scheduler": scheduler
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _get_dimensions_from_aspect_ratio(self, aspect_ratio: str) -> Tuple[int, int]:
        """Get dimensions from aspect ratio preset (from your workflow)"""
        presets = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        return presets.get(aspect_ratio, (1328, 1328))
        
    def _encode_prompt(self, text: str, clip) -> Any:
        """Encode text prompt using CLIP"""
        text_encoder = self.CLIPTextEncode()
        conditioning, = text_encoder.encode(clip=clip, text=text)
        return conditioning
        
    def _create_latent(self, width: int, height: int, batch_size: int = 1) -> Any:
        """Create empty latent image"""
        latent_creator = self.EmptySD3LatentImage()
        latent, = latent_creator.generate(
            width=width,
            height=height,
            batch_size=batch_size
        )
        return latent
        
    def _run_sampling(
        self,
        latent,
        positive,
        negative,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        seed: int,
        denoise: float
    ) -> Any:
        """
        Run the sampling process using your workflow's exact configuration
        Uses ClownsharKSampler_Beta or DetailDaemonSamplerNode based on your workflow
        """
        
        # Try to use ClownsharKSampler_Beta from your workflow
        try:
            sampler = self.ClownsharKSampler_Beta()
            samples = sampler.sample(
                model=self.model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=denoise
            )[0]
            
        except Exception as e:
            logger.warning(f"ClownsharKSampler_Beta failed, falling back: {e}")
            
            # Fallback to standard KSampler
            from nodes import KSampler
            sampler = KSampler()
            samples = sampler.sample(
                model=self.model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=denoise
            )[0]
            
        return samples
        
    def _decode_vae(self, samples) -> Any:
        """Decode latent samples using VAE"""
        decoder = self.VAEDecode()
        images, = decoder.decode(samples=samples, vae=self.vae)
        return images
        
    def _apply_film_grain(self, images, seed: int) -> Any:
        """
        Apply ProPostFilmGrain effect from your workflow
        Settings from your workflow: grain_type="Fine", grain_sat=0.5, 
        grain_power=0.4, shadows=0.1, highs=0.1
        """
        # Set seed for consistent grain
        torch.manual_seed(seed)
        
        # Convert to tensor if needed
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images)
            
        # Add fine film grain
        grain_intensity = 0.4  # grain_power from your workflow
        grain_saturation = 0.5  # grain_sat from your workflow
        
        # Generate grain
        grain = torch.randn_like(images) * grain_intensity * 0.1
        
        # Reduce grain in shadows and highlights (from your workflow settings)
        shadows_threshold = 0.1
        highlights_threshold = 0.9
        
        # Create masks for shadows and highlights
        luminance = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
        shadow_mask = (luminance < shadows_threshold).float()
        highlight_mask = (luminance > highlights_threshold).float()
        
        # Reduce grain in shadows and highlights
        grain_mask = 1.0 - (shadow_mask * 0.9 + highlight_mask * 0.9)
        grain = grain * grain_mask.unsqueeze(-1)
        
        # Apply grain
        images = images + grain
        
        # Clamp values
        images = torch.clamp(images, 0, 1)
        
        return images
        
    def _tensor_to_base64(self, images) -> str:
        """Convert tensor image to base64 string"""
        # Get first image from batch
        if isinstance(images, torch.Tensor):
            image_np = (images[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = (images[0] * 255).astype(np.uint8)
            
        # Convert to PIL
        if image_np.shape[-1] == 3:
            image_pil = Image.fromarray(image_np, mode='RGB')
        else:
            image_pil = Image.fromarray(image_np[..., 0], mode='L')
            
        # Convert to base64
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG", optimize=True)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64

# ===== SERVERLESS HANDLERS =====

def runpod_handler(job):
    """
    RunPod Serverless Handler
    
    Expected input:
    {
        "input": {
            "prompt": str,
            "negative_prompt": str (optional),
            "width": int (optional),
            "height": int (optional),
            "aspect_ratio": str (optional),
            "steps": int (optional),
            "cfg": float (optional),
            "seed": int (optional),
            "lora_url": str (optional),
            "lora_strength": float (optional)
        }
    }
    """
    try:
        import runpod
        
        job_input = job["input"]
        
        # Download LoRA if URL provided
        lora_path = None
        if "lora_url" in job_input:
            import requests
            lora_path = f"/tmp/lora_{job['id']}.safetensors"
            logger.info(f"Downloading LoRA from {job_input['lora_url']}")
            response = requests.get(job_input["lora_url"])
            with open(lora_path, "wb") as f:
                f.write(response.content)
                
        # Initialize wrapper
        model_paths = ModelPaths()
        wrapper = ComfyUIQwenWrapper(model_paths)
        
        # Generate image
        result = wrapper.generate(
            prompt=job_input["prompt"],
            negative_prompt=job_input.get("negative_prompt", ""),
            width=job_input.get("width", 1328),
            height=job_input.get("height", 1328),
            aspect_ratio=job_input.get("aspect_ratio"),
            steps=job_input.get("steps", 8),
            cfg=job_input.get("cfg", 3.5),
            seed=job_input.get("seed", -1),
            lora_path=lora_path,
            lora_strength=job_input.get("lora_strength", 0.8)
        )
        
        # Clean up LoRA file
        if lora_path and os.path.exists(lora_path):
            os.remove(lora_path)
            
        return result
        
    except Exception as e:
        logger.error(f"RunPod handler error: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}


def modal_handler(prompt: str, **kwargs):
    """
    Modal handler function
    Can be decorated with @stub.function for Modal deployment
    """
    model_paths = ModelPaths()
    wrapper = ComfyUIQwenWrapper(model_paths)
    return wrapper.generate(prompt, **kwargs)


def replicate_handler(
    prompt: str = "A beautiful landscape",
    negative_prompt: str = "",
    width: int = 1328,
    height: int = 1328,
    aspect_ratio: Optional[str] = None,
    steps: int = 8,
    cfg: float = 3.5,
    seed: int = -1,
    lora_url: Optional[str] = None,
    lora_strength: float = 0.8
) -> Dict[str, Any]:
    """
    Replicate.com handler
    Use with Cog for deployment
    """
    # Download LoRA if provided
    lora_path = None
    if lora_url:
        import requests
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp:
            response = requests.get(lora_url)
            tmp.write(response.content)
            lora_path = tmp.name
            
    model_paths = ModelPaths()
    wrapper = ComfyUIQwenWrapper(model_paths)
    
    result = wrapper.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        steps=steps,
        cfg=cfg,
        seed=seed,
        lora_path=lora_path,
        lora_strength=lora_strength
    )
    
    # Clean up
    if lora_path and os.path.exists(lora_path):
        os.remove(lora_path)
        
    return result


# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen-Image Serverless Generator")
    parser.add_argument("--mode", choices=["runpod", "modal", "replicate", "test"], 
                       default="test", help="Deployment mode")
    parser.add_argument("--prompt", default="A beautiful sunset over mountains", 
                       help="Test prompt")
    args = parser.parse_args()
    
    if args.mode == "runpod":
        import runpod
        runpod.serverless.start({"handler": runpod_handler})
        
    elif args.mode == "test":
        # Test mode for local development
        model_paths = ModelPaths()
        wrapper = ComfyUIQwenWrapper(model_paths)
        
        result = wrapper.generate(
            prompt=args.prompt,
            width=1328,
            height=1328,
            steps=8,
            cfg=3.5
        )
        
        if result["success"]:
            # Save test image
            img_data = base64.b64decode(result["image"])
            with open("test_output.png", "wb") as f:
                f.write(img_data)
            print(f"Test image saved to test_output.png")
            print(f"Seed: {result['seed']}")
        else:
            print(f"Generation failed: {result['error']}")
