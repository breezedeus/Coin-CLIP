# coding: utf-8

import logging
from typing import List, Union, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.io import read_video

logger = logging.getLogger(__name__)


class BaseModel(object):
    def load_new_checkpoint(self, checkpoint_fp: str) -> None:
        logger.info(f"loading checkpoint from {checkpoint_fp}")
        self.model.load_state_dict(
            torch.load(checkpoint_fp, map_location=self.device)
        )

    def freeze(self):
        """
        冻结模型参数。
        Returns: None

        """
        for param in self.model.parameters():
            param.requires_grad = False

    def get_text_features(self, texts: List[str]) -> np.ndarray:
        """
        抽取文字特征。

        Args:
            texts (list): 待抽取的文字列表

        Returns (np.ndarray): 已归一化后的稠密向量，维度为 [<列表长度>, <向量长度>]

        """
        raise NotImplementedError()

    def get_image_features(
            self, all_imgs: List[Union[str, Path, Image.Image]], img_batch_size=32
    ) -> Tuple[np.ndarray, List[int]]:
        """
        抽取图片特征。

        Args:
            all_imgs (): 待抽取的图片列表
            img_batch_size (): batch size

        Returns:
            第一个值为抽取出的图片特征，为已归一化后的稠密向量，维度为 [<成功抽取的图片数量>, <向量长度>]
            第二个值为成功抽取的图片索引（有些图片因为某些原因会导入失败），长度为 `<成功抽取的图片数量>`

        """
        raise NotImplementedError()

    def get_video_features(
            self, video_path: Union[str, Path], *, num_used_frames=20
    ) -> np.ndarray:
        """
        获得单个视频的向量表示。
        Returns:
            归一化后的一维向量
        """
        logger.info(f"extracting video {video_path}")
        frames, _, _ = read_video(str(video_path), pts_unit="sec", end_pts=10)
        total_frames = frames.shape[0]
        if total_frames < 1:
            raise Exception(f"bad video file {video_path}")
        step = max(total_frames // num_used_frames, 1)
        used_frames = []
        for idx in range(0, total_frames, step):
            used_frames.append(Image.fromarray(frames[idx].numpy()))
        frame_features, _ = self.get_image_features(
            used_frames, img_batch_size=len(used_frames) + 1,
        )
        if len(frame_features) > 0:
            video_feature = np.mean(frame_features, axis=0, keepdims=False)
        else:
            video_feature = np.ones(512)
        return video_feature / (np.linalg.norm(video_feature) + 1e-6)
