import logging
from pathlib import Path
from functools import partial
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import psycopg2
from pgvector.psycopg2 import register_vector

from .dinov2_client import DINOV2Client

class PGVectorClient:
    def __init__(self, images_path: str, _logger: logging.Logger, **credentials):
        self.credentials = credentials
        self.images_path = Path(images_path)
        self.logger = _logger
        self.emdedder = DINOV2Client(logger=self.logger)

    def execute_query(cls, query: str, params: tuple = None, fetch: Callable = None):
        with psycopg2.connect(**cls.credentials) as conn:
            with conn.cursor() as cur:
                register_vector(cur)
                cur.execute(query, params)
                if fetch:
                    return fetch(cur)

    def add_images(self, split: str, filenames: Iterable[str]) -> list[int]:
        query = """
            INSERT INTO images (split, filename)
            VALUES (%s, %s) RETURNING id
        """
        image_ids = []
        for filename in filenames:
            try:
                image_ids.append(self.execute_query(query=query, params=(split, filename), fetch=lambda cur: cur.fetchone()[0]))
            except psycopg2.IntegrityError:
                self.logger.warning(f"Image {filename} already exists in the database")
        return image_ids

    def add_embedding(self, image_id: int, embedding: np.ndarray) -> None:
        query = """
            INSERT INTO image_embeddings (image_id, embedding)
            VALUES (%s, %s)
        """
        self.execute_query(query=query, params=(image_id, embedding))

    def add_embeddings(self, image_ids: list[int]) -> None:
        get_filenames_query = f"""
            SELECT id, filename FROM images WHERE id in ({','.join(map(str, image_ids))})
        """
        filenames = self.execute_query(query=get_filenames_query, fetch=lambda cur: {row[0]: row[1] for row in cur.fetchall()})

        def embedder_callback(image_id, embedding):
            if embedding is not None:
                self.add_embedding(image_id, embedding)

        for image_id, filename in filenames.items():
            image = Image.open(filename)
            self.emdedder(image, partial(embedder_callback, image_id))
