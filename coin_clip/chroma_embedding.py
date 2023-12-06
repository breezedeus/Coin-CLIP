# coding: utf-8
# Copyright (C) 2023, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import importlib
from typing import cast

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.api.types import Document, Embedding, Image

from .coinclip import CoinClip


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
