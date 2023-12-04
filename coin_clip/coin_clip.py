# coding: utf-8
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
        抽取文字特征。

        Args:
            texts (list): 待抽取的文字列表
            max_length (int): token最大长度, CLIP 支持的最大程度是77！
            mode (str): 运行模式，'eval' 或 'train'

        Returns (np.ndarray): 已归一化后的稠密向量，维度为 [<列表长度>, 512]

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
        抽取图片特征。

        Args:
            all_imgs (): 待抽取的图片列表
            img_batch_size (): batch size
            mode (str): 运行模式，'eval' 或 'train'

        Returns:
            第一个值为抽取出的图片特征，为已归一化后的稠密向量，维度为 [<成功抽取的图片数量>, 512]
            第二个值为成功抽取的图片索引（有些图片因为某些原因会导入失败），长度为 `<成功抽取的图片数量>`

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
