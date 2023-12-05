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
        Freeze all model parameters.
        Returns: None

        """
        for param in self.model.parameters():
            param.requires_grad = False

    def get_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract Text Features.

        Args:
            texts (list): List of text to be extracted.

        Returns:
            (np.ndarray): Normalized dense vectors, with dimensions [<Length of the List>, <Vector Length>].
        """
        raise NotImplementedError()

    def get_image_features(
            self, all_imgs: List[Union[str, Path, Image.Image]], img_batch_size=32
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract Image Features.

        Args:
            all_imgs (): List of images to be extracted.
            img_batch_size (): Batch size.

        Returns:
            The first value is the extracted image features, which are normalized dense vectors, with dimensions [<Number of Successfully Extracted Images>, <Vector Length>].
            The second value is the indices of successfully extracted images (some images may fail to import for certain reasons), with a length of `<Number of Successfully Extracted Images>`.
        """
        raise NotImplementedError()

    def get_video_features(
            self, video_path: Union[str, Path], *, num_used_frames=20
    ) -> np.ndarray:
        """
        Obtain the vector representation for a single video.
        Returns:
            Normalized one-dimensional vector.
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
