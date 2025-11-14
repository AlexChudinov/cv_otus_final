import requests
from io import BytesIO
from itertools import islice
from threading import Lock, Event

import PIL.Image as PILImage
from PIL.Image import Image as PillowImage

from .constants import LOGGER_FORMAT

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
    return logger
