from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .utils import nms

class ImageProcessor(ABC):
    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess(self, predictions: np.ndarray) -> np.ndarray:
        pass

class YOLOv8Processor(ImageProcessor):
    _MEAN = np.array([0.485, 0.456, 0.406])
    _STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, model_w, model_h, orig_w, orig_h, nms_iou=0.5):
        self.model_w = model_w
        self.model_h = model_h
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.transform = transforms.Compose([transforms.Resize((self.model_h, self.model_w)),
                                             transforms.Normalize(mean=self._MEAN, std=self._STD)])
        self.nms_iou = nms_iou

    def estimate_dims(self) -> None:
        self.model_aspect_ratio = self.model_w / self.model_h
        self.orig_aspect_ratio = self.orig_w / self.orig_h
        if self.model_aspect_ratio > self.orig_aspect_ratio:
            self.new_w = int(round(self.orig_h * self.model_aspect_ratio))
            self.new_h = self.orig_h
        else:
            self.new_w = self.orig_w
            self.new_h = int(round(self.orig_w / self.model_aspect_ratio))

        self.dw = (self.new_w - self.orig_w) // 2
        self.dh = (self.new_h - self.orig_h) // 2

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        self.estimate_dims()
        output = torch.from_numpy(self._MEAN[..., None, None] * np.ones((3, self.new_h, self.new_w)))
        output[:, self.dh:self.dh + self.orig_h, self.dw:self.dw + self.orig_w] = transforms.ToTensor()(image)
        output = self.transform(output)
        return output

    def postprocess(self, predictions: np.ndarray) -> np.ndarray:
        x, y, w, h = predictions[:4, ...]
        scores = predictions[4:, ...].T
        x = (x - self.dw) / (self.model_w - self.dw * 2) * self.orig_w
        y = (y - self.dh) / (self.model_h - self.dh * 2) * self.orig_h
        w = w / (self.model_w - self.dw * 2) * self.orig_w
        h = h / (self.model_h - self.dh * 2) * self.orig_h

        nms_filtered_ids = nms(x - w/2, y - h/2, x + w/2, y + h/2, scores.max(-1), scores.argmax(-1), self.nms_iou)

        return np.concat((x[nms_filtered_ids, None],
                          y[nms_filtered_ids, None],
                          w[nms_filtered_ids, None],
                          h[nms_filtered_ids, None],
                          scores[nms_filtered_ids, ...]), axis=-1)
