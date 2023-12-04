# coding: utf-8
import importlib
from typing import cast

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.api.types import Document, Embedding, Image

from .coin_clip import CoinClip


class ChromaEmbeddingFunction(OpenCLIPEmbeddingFunction):
    def __init__(
        self,
        model_name='breezedeus/coin-clip-vit-base-patch32',
        device='cpu',
        max_image_size=2048,
        **kwargs,
    ):
        try:
            self._PILImage = importlib.import_module("PIL.Image")
        except ImportError:
            raise ValueError(
                "The PIL python package is not installed. Please install it with `pip install pillow`"
            )

        self.coin_clip = CoinClip(model_name, device, max_image_size, **kwargs)

    def _encode_image(self, image: Image) -> Embedding:
        pil_image = self._PILImage.fromarray(image)
        img_features, success_ids = self.coin_clip.get_image_features([pil_image])
        if len(success_ids) == 0:
            raise ValueError("Failed to get image features")
        return cast(Embedding, img_features[0, :].tolist())

    def _encode_text(self, text: Document) -> Embedding:
        text_features = self.coin_clip.get_text_features([text])
        return cast(Embedding, text_features[0, :].tolist())
