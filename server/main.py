import gradio as gr
import torch
from diffusers import LatentConsistencyModelImg2ImgPipeline
from PIL import Image, ImageChops

WIDTH = 256
HEIGHT = 256


class LCM:
    def __init__(
        self,
        model_id: str = "SimianLuo/LCM_Dreamshaper_v7",
        initial_prompt: str = "psychedelic",
    ):
        self.pipeline = self._load_pipeline(model_id)
        self._empty_image = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color="white")
        self._prompt = initial_prompt
        self._prev_sketch_image = self._empty_image.copy()
        self._prev_progressive_image = self._empty_image.copy()

    @property
    def prompt(self) -> str:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def get_sketch(
        self, progressive_image: Image.Image, sketch_image: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        self._prev_sketch_image = sketch_image
        sketch_image = (
            ImageChops.difference(sketch_image, self._prev_sketch_image)
            .resize((WIDTH, HEIGHT))
            .convert("L")
            .point(lambda x: 0 if x < 30 else 1, "1")
        )
        progressive_image = progressive_image.resize((WIDTH, HEIGHT)).convert("RGBA")
        progressive_image.paste(sketch_image, (0, 0), sketch_image)
        self._prev_progressive_image = progressive_image

        return self._prev_progressive_image

    def progress(self) -> Image.Image:
        try:
            self._prev_progressive_image = self.pipeline(
                prompt=self.prompt,
                image=self._prev_progressive_image.resize((WIDTH, HEIGHT)),
                num_inference_steps=1,
                strength=0.5,
                width=WIDTH,
                height=HEIGHT,
                guidance_scale=8.0,
                lcm_origin_steps=10,
                output_type="pil",
            ).images[0]
        except Exception:
            pass

        return self._prev_progressive_image

    def _load_pipeline(self, model_id: str):
        pipeline = LatentConsistencyModelImg2ImgPipeline.from_pretrained(model_id)
        pipeline.safety_checker = None
        pipeline.to(device="cuda", dtype=torch.float16)
        return pipeline


lcm = LCM()
with gr.Blocks() as ui:
    prompt = gr.Textbox(label="prompt", value=lcm.prompt)
    with gr.Row():
        sketch = gr.Image(
            value=lcm._empty_image,
            tool="color-sketch",
            image_mode="RGB",
            type="pil",
            width=WIDTH,
            height=HEIGHT,
        )
        image = gr.Image(
            value=lcm.progress,
            image_mode="RGB",
            type="pil",
            width=WIDTH,
            height=HEIGHT,
            every=1,
        )

    sketch.change(
        fn=lcm.get_sketch,
        inputs=[image, sketch],
        outputs=[image],
        show_progress="hidden",
    )

    prompt.change(
        fn=lambda x: lcm.__setattr__("prompt", x),
        inputs=[prompt],
        outputs=None,
        show_progress="hidden",
    )

ui.queue().launch()
