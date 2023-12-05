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

import logging
from typing import List, Union, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel

from .utils import read_img
from .base_model import BaseModel

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CoinClip(BaseModel):
    """
    Ref: https://huggingface.co/docs/transformers/model_doc/clip
    """

    def __init__(
        self,
        model_name='breezedeus/coin-clip-vit-base-patch32',
        device='cpu',
        max_image_size=2048,
        **kwargs,
    ):
        """

        Args:
            model_name (str): model name; either local path or huggingface model name
            device (str): device; either 'cpu' or 'cuda'
            max_image_size (int): max image size for feature extraction
            **kwargs ():
        """
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device=device)
        self.model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.max_image_size = max_image_size

    def get_text_features(
        self, texts: List[str], max_length: int = 77, mode: str = 'eval'
    ) -> np.ndarray:
        """
        Extract text features.

        Args:
            texts (list): List of text to be extracted.
            max_length (int): Maximum token length; CLIP supports a maximum of 77!.
            mode (str): Execution mode, 'eval' or 'train'.

        Returns:
            (np.ndarray): Normalized dense vectors, with dimensions [<Length of the List>, 512].
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(device=self.device)
        if mode != 'train':
            self.model.eval()
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                return F.normalize(text_features, dim=1).cpu().detach().numpy()
        else:
            self.model.train()
            text_features = self.model.get_text_features(**inputs)
            return F.normalize(text_features, dim=1)

    def get_image_features(
        self,
        all_imgs: List[Union[str, Path, Image.Image]],
        img_batch_size=32,
        mode: str = 'eval',
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract image features.

        Args:
            all_imgs (): List of images to be extracted.
            img_batch_size (): Batch size.
            mode (str): Execution mode, 'eval' or 'train'.

        Returns:
            The first value is the extracted image features, which are normalized dense vectors, with dimensions [<Number of Successfully Extracted Images>, 512].
            The second value is the indices of successfully extracted images (some images may fail to import for certain reasons), with a length of `<Number of Successfully Extracted Images>`.
        """
        img_features = []
        success_ids = []

        for i in range((len(all_imgs) // img_batch_size) + 1):
            logger.debug(f"extracting image {i * img_batch_size}")
            imgs = []
            for _i in range(i * img_batch_size, (i + 1) * img_batch_size):
                if _i >= len(all_imgs):
                    break
                img = all_imgs[_i]
                try:
                    _img = read_img(img) if not isinstance(img, Image.Image) else img
                    if (r := self.max_image_size / max(_img.size)) < 1:
                        _img = _img.resize((np.array(_img.size) * r).astype(int))
                    imgs.append(_img)
                    success_ids.append(_i)
                except:
                    continue

            if len(imgs) > 0:
                inputs = self.processor(images=imgs, return_tensors="pt").to(
                    device=self.device
                )
                if mode != 'train':
                    self.model.eval()
                    with torch.no_grad():
                        _image_features = self.model.get_image_features(**inputs)
                else:
                    self.model.train()
                    _image_features = self.model.get_image_features(**inputs)
                # _image_features = F.normalize(_image_features, dim=1)
                if mode != 'train':
                    img_features.append(_image_features.cpu().detach())
                else:
                    img_features.append(_image_features)
        img_features = torch.cat(img_features)
        if not img_features.is_cuda:
            img_features = img_features.float()
        img_features = F.normalize(img_features, dim=1)
        if mode != 'train':
            img_features = img_features.numpy()
        return img_features, success_ids


if __name__ == '__main__':
    model = CoinClip()
    texts = ['a black clothes', 'a white bathroom']
    images = ['examples/10_back.jpg', 'examples/16_back.jpg']
    txt_feats = model.get_text_features(texts)
    img_feats, success_ids = model.get_image_features(images)
