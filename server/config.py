from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # General
    target_resolution: int = 640

    # Server
    host: str = "0.0.0.0"
    port: int = 9090
    workers: int = 1

    # Acceleration
    torch_compile: bool = True
    xformers: bool = False
    device: Literal["cpu", "cuda"] = "cuda"

    # Prompt
    prompt_model_id: str = "Gustavosta/MagicPrompt-Stable-Diffusion"

    # LCM
    lcm_model_id: str = "SimianLuo/LCM_Dreamshaper_v7"
    dtype: Literal["float32", "float16"] = "float16"
    generation_resolution: int = 320
    initial_prompt: str = "psychedelic structure, high quality"
    negative_prompt: str = "jpeg artifacts, low quality, bad quality, bad compression, low resolution, blurry"
    inference_steps: int = 2
    strength: float = 0.35
    guidance_scale: float = 8.0
    original_inference_steps: int = 12

    # Super Resolution
    use_super_resolution: bool = True
    superres_scale: int = 2
    realesrgan_model_path_format: str = "/app/models/RealESRGAN_x{}.pth"

    def __post_init__(self):
        if self.xformers and self.torch_compile:
            raise ValueError("xformers and torch_compile cannot be both True")
