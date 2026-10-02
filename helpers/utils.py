from io import BytesIO
import base64
import math
import yaml
import random
import numpy as np

import torch
from easydict import EasyDict as edict
from PIL import Image

@torch.no_grad()
def decode_latents(
    pipeline,
    latents: torch.Tensor
):
    """
    Decode latents to image.

    Args:
        pipeline: The diffusion pipeline.
        latents: The latents to decode.
    """
    if len(latents.shape) == 3:
        latents_for_decode = latents.unsqueeze(0).to(pipeline.vae.dtype)
    elif len(latents.shape) == 4:
        latents_for_decode = latents.to(pipeline.vae.dtype)
    else:
        raise ValueError(f"Invalid latent shape: {latents.shape}")
    unscaled_latents = (latents_for_decode / pipeline.vae.config.scaling_factor) \
        + pipeline.vae.config.shift_factor
    decoded_tensors = pipeline.vae.decode(unscaled_latents, return_dict=False)[0]
    return pipeline.image_processor.postprocess(decoded_tensors, output_type="pil")

def load_config(config_path: str) -> edict:
    """
    Load config from yaml file.

    Args:
        config_path: The path to the config file.
    
    Returns:
        edict: The config as an EasyDict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return edict(config)

def pil_to_base64(img: Image.Image) -> str:
    """
    Convert PIL image to base64 string.
    
    Args:
        img: The PIL image.
        
    Returns:
        str: The base64 string.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def barker_acceptance_prob(delta_D: float, temperature: float) -> float:
    """
    Calculate the Barker acceptance probability.
    
    Args:
        delta_D: The change in distance.
        temperature: The temperature.
        
    Returns:
        float: The acceptance probability.
    """
    try:
        # Clamp to [-700, 700] to prevent math.exp() OverflowError if the model hallucinates wildly
        exponent = max(-700.0, min(700.0, delta_D / temperature))
        p_accept = 1.0 / (1.0 + math.exp(exponent))
    except OverflowError:
        p_accept = 0.0
    return p_accept

def metropolis_acceptance_prob(delta_D: float, temperature: float) -> float:
    """
    Calculate the Metropolis-Hastings acceptance probability.
    
    Args:
        delta_D: The change in distance.
        temperature: The temperature.
        
    Returns:
        float: The acceptance probability.
    """
    if delta_D < 0:
        # Metropolis strictly accepts ALL improvements (100% probability)
        p_accept = 1.0 
    else:
        # For regressions, calculate the standard Metropolis decay
        try:
            # delta_D is positive here. Note the negative sign in the exponent!
            exponent = max(-700.0, min(0.0, -delta_D / temperature))
            p_accept = math.exp(exponent)
        except OverflowError:
            p_accept = 0.0
    return p_accept

def hill_climb_acceptance_prob(delta_D: float, temperature: float = None) -> float:
    """
    Calculate the Hill Climb acceptance probability (strict greedy).
    
    Args:
        delta_D: The change in distance.
        temperature: Unused, kept for signature compatibility.
        
    Returns:
        float: 1.0 if distance decreased (improved), else 0.0.
    """
    return 1.0 if delta_D < 0 else 0.0

def set_seed(seed: int, generator: torch.Generator = None):
    """
    Set random seeds across Python, PyTorch, and optional Generator for reproducibility.
    
    Args:
        seed: The seed integer.
        generator: An optional torch.Generator to seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if generator is not None:
        generator.manual_seed(seed)
