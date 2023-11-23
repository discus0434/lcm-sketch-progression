import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from lcm_sketch_progression import Progressor

logger = logging.getLogger("uvicorn")
PROJECT_DIR = Path(__file__).parent.parent


class PredictInputModel(BaseModel):
    """
    The input model for the /predict endpoint.
    """

    base64_image: str


class PredictResponseModel(BaseModel):
    """
    The response model for the /predict endpoint.
    """

    base64_image: str


class UpdatePromptResponseModel(BaseModel):
    """
    The response model for the /update_prompt endpoint.
    """

    prompt: str


class Api:
    def __init__(self, **config) -> None:
        """
        Initialize the API.

        Parameters
        ----------
        config : dict
            The configuration.
        """
        self.progressor = Progressor(**config)
        self.app = FastAPI()
        self.app.add_api_route(
            "/predict",
            self._predict,
            methods=["POST"],
            response_model=PredictResponseModel,
        )
        self.app.add_api_route(
            "/update_prompt",
            self._update_prompt,
            methods=["GET"],
            response_model=UpdatePromptResponseModel,
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._predict_lock = asyncio.Lock()
        self._update_prompt_lock = asyncio.Lock()

    async def _predict(self, inp: PredictInputModel) -> PredictResponseModel:
        """
        Predict whether the text content is safe or not.

        Parameters
        ----------
        inp : InputModel
            The input model.

        Returns
        -------
        PredictResponseModel
            The prediction result.
        """
        async with self._predict_lock:
            return PredictResponseModel(
                base64_image=self._pil_to_base64(
                    self.progressor.progress(self._base64_to_pil(inp.base64_image))
                )
            )

    async def _update_prompt(self) -> UpdatePromptResponseModel:
        """
        Update the prompt and return the updated prompt.

        Returns
        -------
        UpdatePromptResponseModel
            The updated prompt.
        """
        async with self._update_prompt_lock:
            self.progressor.update_prompt()
            return UpdatePromptResponseModel(prompt=self.progressor.prompt)

    def _pil_to_base64(self, image: Image.Image, format: str = "JPEG") -> bytes:
        """
        Convert a PIL image to base64.

        Parameters
        ----------
        image : Image.Image
            The PIL image.

        format : str
            The image format, by default "JPEG".

        Returns
        -------
        bytes
            The base64 image.
        """
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    def _base64_to_pil(self, base64_image: str) -> Image.Image:
        """
        Convert a base64 image to PIL.

        Parameters
        ----------
        base64_image : str
            The base64 image.

        Returns
        -------
        Image.Image
            The PIL image.
        """
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]
        return Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB")


if __name__ == "__main__":
    from config import Config

    config = Config()

    uvicorn.run(
        Api(**config.__dict__).app,
        host=config.host,
        port=config.port,
        workers=config.workers,
    )
