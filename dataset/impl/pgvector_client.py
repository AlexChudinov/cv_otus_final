import logging
from pathlib import Path
from functools import partial
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import psycopg2
from pgvector.psycopg2 import register_vector

from .dinov2_client import DINOV2Client
from .utils import CallbackManager, download_image

class PGVectorClient:
    SIMILARITY_FUNCTIONS = {
        "cosine": "<=>",
        "dot": "<#>",
        "L2": "<->",
        "L1": "<+>",
    }

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

    def filter_file_exist(self, filenames: list[str | Path]) -> list[str]:
        filenames = [f"'{filename}'" for filename in filenames]
        query = f"""
            SELECT filename FROM images WHERE filename IN ({','.join(filenames)})
        """
        return self.execute_query(query=query, fetch=lambda cur: [row[0] for row in cur.fetchall()])

    def filter_file_not_exist(self, filenames: list[str | Path]) -> list[str]:
        existing_filenames = set(self.filter_file_exist(filenames))
        return [filename for filename in map(str, filenames) if filename not in existing_filenames]

    def filter_embedding_exist(self, image_ids: list[int]) -> list[int]:
        query = f"""
            SELECT image_id FROM image_embeddings WHERE image_id in ({','.join(map(str, image_ids))})
        """
        return self.execute_query(query=query, fetch=lambda cur: [row[0] for row in cur.fetchall()])

    def filter_embedding_not_exist(self, image_ids: list[int]) -> list[int]:
        existing_image_ids = set(self.filter_embedding_exist(image_ids))
        return [image_id for image_id in image_ids if image_id not in existing_image_ids]

    def not_encoded_id_chunked_iter(self, chunk_size=5000) -> Iterable[list[int]]:
        min_id = 0
        while True:
            query = f"SELECT id FROM images WHERE id > {min_id} ORDER BY id ASC LIMIT {chunk_size}"
            next_chunk = self.execute_query(query=query, fetch=lambda cur: [row[0] for row in cur.fetchall()])
            if not next_chunk:
                break

            min_id = next_chunk[-1]
            next_chunk = self.filter_embedding_not_exist(next_chunk)
            if not next_chunk:
                continue

            yield next_chunk

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
        callback_manager = CallbackManager()
        def embedder_callback(image_id, embedding):
            if embedding is not None:
                self.add_embedding(image_id, embedding)
            callback_manager.done()

        for image_id, filename in filenames.items():
            image = Image.open(filename)
            callback_manager.add()
            embedder_run = partial(self.emdedder, image, partial(embedder_callback, image_id))

        embedder_run()
        callback_manager.wait()

    def get_neighbors(self, query_image_url: str, k: int = 10, similarity: str = "cosine") -> list[tuple[float, str]]:
        image = download_image(query_image_url)
        callback_manager = CallbackManager()

        callback_result = []
        def embedder_callback(embedding):
            callback_result.append(embedding)
            callback_manager.done()

        callback_manager.add()
        self.emdedder(image, embedder_callback)
        callback_manager.wait()

        query_embedding, *_ = callback_result
        assert query_embedding is not None, f"Failed to build embedding for image url: {query_image_url}"

        query = f"""
            SELECT %s {self.SIMILARITY_FUNCTIONS[similarity]} embedding AS similarity, filename FROM image_embeddings
            JOIN images ON image_embeddings.image_id = images.id ORDER BY similarity DESC LIMIT %s
        """
        return self.execute_query(query=query, params=(query_embedding, k), fetch=lambda cur: [(row[0], row[1]) for row in cur.fetchall()])
