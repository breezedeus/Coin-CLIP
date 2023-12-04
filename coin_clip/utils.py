# utf-8
#
import logging
import urllib
import pathlib
from pathlib import Path
from typing import Union

import colorlog
from PIL import Image, ImageOps
import numpy as np
import torch

from .consts import IMAGE_SUFFIX, VIDEO_SUFFIX


logging.captureWarnings(True)
logger = logging.getLogger()


def set_logger(level=logging.INFO):
    logger.setLevel(level)

    # 创建一个带有颜色的日志记录器
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s %(asctime)s %(funcName)s:%(lineno)d %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
        )
    )
    logger.addHandler(handler)
    return logger


def guess_type_from_url(url):
    suffix = pathlib.Path(urllib.parse.urlparse(url.lower()).path).suffix[1:]
    if suffix == 'gif':
        return 'gif'
    elif suffix in VIDEO_SUFFIX:
        return 'video'
    elif suffix in IMAGE_SUFFIX:
        return 'image'
    else:
        return None


def read_img(path: Union[str, Path, Image.Image]) -> Image.Image:
    """

    Args:
        path (str): image file path

    Returns: RGB Image.Image
    """
    img = Image.open(path) if not isinstance(path, Image.Image) else path
    img = ImageOps.exif_transpose(img).convert('RGB')  # 识别旋转后的图片（pillow不会自动识别）
    return img


def resize_img(img: Image.Image, output_size: int) -> Image.Image:
    """

    Args:
        img (Image.Image): Image to be resized.
        output_size (int): Maximum size of the image.

    Returns: resized Image.Image
    """
    w, h = img.size
    if max(w, h) > output_size:
        if w > h:
            w, h = output_size, int(output_size * h / w)
        else:
            w, h = int(output_size * w / h), output_size
        img = img.resize((w, h), Image.BILINEAR)
    return img


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )
