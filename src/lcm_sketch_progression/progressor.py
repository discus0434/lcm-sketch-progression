import random

import torch
from diffusers import LatentConsistencyModelImg2ImgPipeline
from PIL import Image
from RealESRGAN import RealESRGAN
from transformers import pipeline


class Progressor:
    def __init__(self, **config):
        self.config = config
        self.target_resolution = self.config["target_resolution"]
        self.generation_resolution = self.config["generation_resolution"]

        self._empty_image = Image.new(
            mode="RGB",
            size=(self.generation_resolution,) * 2,
            color="white",
        )
        self.prompt = self.config["initial_prompt"]

        self.lcm_pipeline = self._load_lcm_pipeline()
        self.esrgan_model = self._load_esrgan_model()
        self.prompt_pipeline = self._load_prompt_pipeline()

    def progress(self, progressive_image: Image.Image) -> Image.Image:
        progressive_image = self.lcm_pipeline(
            prompt=self.prompt,
            negative_prompt=self.config["negative_prompt"],
            image=progressive_image.resize(
                (self.generation_resolution,) * 2, Image.Resampling.LANCZOS
            ),
            num_inference_steps=self.config["inference_steps"],
            strength=self.config["strength"],
            width=self.generation_resolution,
            height=self.generation_resolution,
            guidance_scale=self.config["guidance_scale"],
            original_inference_steps=self.config["original_inference_steps"],
            output_type="pil",
        ).images[0]

        if self.config["use_super_resolution"]:
            return self.esrgan_model.predict(progressive_image)
        else:
            return progressive_image.resize(
                (self.target_resolution,) * 2, Image.Resampling.LANCZOS
            )

    def update_prompt(self) -> None:
        self.prompt = self.prompt_pipeline(
            random.choice(self._prefixs), max_length=random.randint(10, 25)
        )[0]["generated_text"].strip()

    def _load_lcm_pipeline(self) -> LatentConsistencyModelImg2ImgPipeline:
        lcm_pipeline = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
            self.config["lcm_model_id"],
            safety_checker=None,
        )
        lcm_pipeline.to(
            device=self.config["device"],
            dtype=torch.float16 if self.config["dtype"] == "float16" else torch.float32,
        )
        lcm_pipeline.unet.to(memory_format=torch.channels_last)

        if self.config["torch_compile"]:
            lcm_pipeline.unet = torch.compile(
                lcm_pipeline.unet, mode="reduce-overhead", fullgraph=True
            )
            lcm_pipeline.vae = torch.compile(
                lcm_pipeline.vae, mode="reduce-overhead", fullgraph=True
            )

        if self.config["xformers"]:
            lcm_pipeline.enable_xformers_memory_efficient_attention()

        # Warmup
        lcm_pipeline(
            prompt="warmup",
            image=self._empty_image,
            num_inference_steps=1,
            guidance_scale=8.0,
        )

        return lcm_pipeline

    def _load_esrgan_model(self) -> RealESRGAN | None:
        esrgan_model = None
        if self.config["use_super_resolution"]:
            esrgan_model = RealESRGAN(
                device=self.config["device"],
                scale=self.config["superres_scale"],
            )
            esrgan_model.load_weights(
                self.config["realesrgan_model_path_format"].format(
                    self.config["superres_scale"]
                ),
                download=True,
            )

            if self.config["torch_compile"]:
                esrgan_model.model = torch.compile(
                    esrgan_model.model, mode="reduce-overhead", fullgraph=True
                )

            # Warmup
            esrgan_model.predict(self._empty_image)

        return esrgan_model

    def _load_prompt_pipeline(self):
        prompt_pipeline = pipeline(
            "text-generation",
            model=self.config["prompt_model_id"],
            tokenizer="gpt2",
        )

        return prompt_pipeline

    @property
    def _prefixs(self) -> list[str]:
        return [
            "psychedelic structure of",
            "landscape of",
            "fantasic",
            "realistic",
            "abstract",
            "futuristic",
            "1girl",
            "1boy",
            "realistic",
            "anime",
        ]
