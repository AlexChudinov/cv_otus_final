import hashlib
import requests
from io import BytesIO
from itertools import islice
from threading import Lock, Event

import numpy as np
import PIL.Image as PILImage
from PIL.Image import Image as PillowImage

from .constants import LOG_LEVEL, LOGGER_FORMAT

def chunked_iter(iterable, chunk_size=100):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            return
        yield chunk

def download_image(image_url: str) -> PillowImage:
    r = requests.get(image_url)
    r.raise_for_status()
    return PILImage.open(BytesIO(r.content))

class CallbackManager:
    def __init__(self):
        self._lock = Lock()
        self._event = Event()
        self._tasks_count = 0

    def add(self):
        with self._lock:
            self._tasks_count += 1
            self._event.clear()

    def done(self):
        with self._lock:
            self._tasks_count -= 1
            if self._tasks_count == 0:
                self._event.set()

    def wait(self):
        self._event.wait()

def create_logger():
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format=LOGGER_FORMAT)
    logger.setLevel(LOG_LEVEL)
    return logger

def cross_iou_matrix(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    xmin = np.maximum(x1[..., None], x1[None, ...])
    ymin = np.maximum(y1[..., None], y1[None, ...])
    xmax = np.minimum(x2[..., None], x2[None, ...])
    ymax = np.minimum(y2[..., None], y2[None, ...])
    i = np.maximum(0.0, xmax - xmin) * np.maximum(0.0, ymax - ymin)
    u = (x2 - x1) * (y2 - y1)
    return i / (u[..., None] + u[None, ...] - i)

def nms(x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        nms_threshold: float) -> list[int]:
    cross_iou = cross_iou_matrix(x1, y1, x2, y2)
    visited = set()
    result = []
    for i, class_id in enumerate(class_ids):
        if i in visited:
            continue

        new_ids, *_ = np.where((class_ids == class_id) & (cross_iou[i, ...] >= nms_threshold))
        visited.update(new_ids)
        result.append(new_ids[scores[new_ids].argmax()])

    return sorted(result, key=lambda i: scores[i], reverse=True)

def rgb_color_from_id(id: int) -> tuple[int, int, int]:
    hash = hashlib.md5(str(id).encode()).hexdigest()[:6]
    return tuple(map((lambda x: int(x, 16)), (hash[0:2], hash[2:4], hash[4:6])))
