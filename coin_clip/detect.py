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

from typing import List, Dict, Any

import torch
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from .utils import read_img


class Detector(object):
    def __init__(
        self, model_name="google/owlvit-base-patch32", object_names='coin', device="cpu"
    ):
        device = torch.device(device)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.object_names = object_names.split(",")
        self.device = device

    @torch.no_grad()
    def detect(
        self, img: np.ndarray, score_threshold: float = 0.1, **kwargs,
    ) -> List[Dict[str, Any]]:
        """

        Args:
            img (): np.ndarray, [H, W, C], RGB-format
            score_threshold (float, optional): Defaults to 0.1.

        Returns: list, with each element like this, box uses relative coordinates, [xmin, ymin, xmax, ymax]
            {'box': [x1, y1, x2, y2], 'score': score, 'label': label}

        """
        ori_h, ori_w = img.shape[0], img.shape[1]
        target_sizes = torch.Tensor([img.shape[:2]])  # -> [H, W]

        inputs = self.processor(
            text=self.object_names, images=img, return_tensors="pt", **kwargs
        ).to(self.device)
        outputs = self.model(**inputs)

        outputs.logits = outputs.logits.cpu()
        outputs.pred_boxes = outputs.pred_boxes.cpu()
        results = self.processor.post_process(
            outputs=outputs, target_sizes=target_sizes
        )
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"],
        )
        outs = []
        for box, score, label in zip(boxes, scores, labels):
            if score < score_threshold:
                continue
            # 返回相对坐标
            box = list(
                map(
                    float,
                    [box[0] / ori_w, box[1] / ori_h, box[2] / ori_w, box[3] / ori_h],
                )
            )
            outs.append({'box': box, 'score': float(score), 'label': int(label)})

        outs.sort(key=lambda x: x['score'], reverse=True)
        return outs


def visualize_result(img, outs, object_names, show=False):
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    ori_h, ori_w = img.shape[:2]
    img = img.copy()

    for out in outs:
        box, score, label = out['box'], out['score'], out['label']
        box = [
            int(ori_w * box[0]),
            int(ori_h * box[1]),
            int(ori_w * box[2]),
            int(ori_h * box[3]),
        ]

        img = cv2.rectangle(img, box[:2], box[2:], (255, 0, 0), 5)
        if box[3] + 25 > 768:
            y = box[3] - 10
        else:
            y = box[3] + 25

        img = cv2.putText(
            img, object_names[label], (box[0], y), font, 1, (255, 0, 0), 2, cv2.LINE_AA,
        )
    if show:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return img


if __name__ == '__main__':
    import time

    model_name = "google/owlvit-base-patch32"
    img_path = 'docs/c4.jpg'
    img = read_img(img_path)
    w, h = img.size
    img = img.resize((w // 2, h // 2))
    img = np.array(img)

    # resized = {"height": h // 2, "width": w // 2}
    detector = Detector(model_name=model_name)
    time_costs = []
    for i in range(1):
        start_time = time.time()
        outs = detector.detect(img, score_threshold=0.1)
        time_costs.append(time.time() - start_time)

    print('avg time cost: ', np.mean(time_costs))
    print(f'find {len(outs)} objects: ', outs)

    visualize_result(img, outs, detector.object_names, show=True)
