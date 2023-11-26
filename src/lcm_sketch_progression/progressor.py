import random

import torch
from diffusers import AutoencoderTiny, LatentConsistencyModelImg2ImgPipeline
from PIL import Image
from RealESRGAN import RealESRGAN
from transformers import pipeline


class Progressor:
    def __init__(self, **config):
        """
        Initialize the progressor.

        Parameters
        ----------
        config : dict
            The configuration.
        """
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
        """
        Progress the generation.

        Parameters
        ----------
        progressive_image : Image.Image
            The image to be progressed.

        Returns
        -------
        Image.Image
            The progressed image.
        """
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
        """
        Update the prompt with generating words brought up by the
        random prefix.
        """
        self.prompt = self.prompt_pipeline(
            random.choice(self._prefixes), max_length=random.randint(10, 25)
        )[0]["generated_text"].strip()

    def _load_lcm_pipeline(self) -> LatentConsistencyModelImg2ImgPipeline:
        """
        Load the LCM pipeline.

        To optimize the performance, the LCM pipeline will be:

        1. inferred with float16
        2.a. compiled with `torch.compile` if `torch_compile` is True
        2.b. enabled with `xformers` if `xformers` is True
        3. warmed up with an empty image

        Returns
        -------
        LatentConsistencyModelImg2ImgPipeline
            The LCM pipeline.
        """
        lcm_pipeline = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
            self.config["lcm_model_id"],
            safety_checker=None,            
            feature_extractor=None,
        )
        lcm_pipeline.vae = AutoencoderTiny.from_pretrained(self.config["vae_model_id"])
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

        for _ in range(3):
            # Warmup
            lcm_pipeline(
                prompt="warmup",
                image=self._empty_image,
                num_inference_steps=1,
                guidance_scale=8.0,
            )

        return lcm_pipeline

    def _load_esrgan_model(self) -> RealESRGAN | None:
        """
        Load the ESRGAN model.

        To optimize the performance, the ESRGAN model will be compiled
        with `torch.compile` if `torch_compile` is True.

        Returns
        -------
        RealESRGAN | None
            The ESRGAN model. If super resolution is not used, this will
            be None.
        """
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
        """
        Load the prompt pipeline.

        Returns
        -------
        pipeline
            The prompt pipeline.
        """
        prompt_pipeline = pipeline(
            "text-generation",
            model=self.config["prompt_model_id"],
            tokenizer="gpt2",
        )

        return prompt_pipeline

    @property
    def _prefixes(self) -> list[str]:
        """
        The prefixes for the prompt.
        """
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
