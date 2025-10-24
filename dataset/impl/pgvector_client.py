import logging
import requests
import uuid

import psycopg2

from io import BytesIO
from pathlib import Path
from typing import Callable

from pgvector.psycopg2 import register_vector
from PIL import Image

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

class PGVectorClient:
    def __init__(self, images_path: str, observers: list[Callable[[str], None]] = None, **credentials):
        self.credentials = credentials
        self.images_path = Path(images_path)
        self.observers = observers if observers else []

    def execute_query(cls, query: str, params: tuple = None, fetch: Callable = None):
        with psycopg2.connect(**cls.credentials) as conn:
            with conn.cursor() as cur:
                register_vector(cur)
                cur.execute(query, params)
                if fetch:
                    return fetch(cur)

    def add_images(self, urls: list[str]) -> None:
        query = """
            INSERT INTO images (url, filename)
            VALUES (%s, %s)
            ON CONFLICT (url) DO NOTHING;
        """
        for url in urls:
            filename = f"{uuid.uuid4()}.jpg"
            img = download_image(url)
            with (self.images_path / filename).open("w") as f:
                img.save(f)

            self.execute_query(
                query=query,
                params=(url, filename)
            )
            _logger.info("action=%s\timage_url=%s\timage_filename=%s\turl='%s'", "load image", url, filename, url)

            for observer in self.observers:
                observer(url)

    def get_summary_info(self) -> list:
        query = """
            SELECT * FROM images
        """
        return self.execute_query(query=query, fetch=lambda cur: cur.fetchall())
