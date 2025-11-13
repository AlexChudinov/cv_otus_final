import logging

from .constants import IMAGES_PATH, LOGGER_FORMAT, PGVECTOR_CREDENTIALS
from .pgvector_client import PGVectorClient


logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def encoder_images():
    client = PGVectorClient(IMAGES_PATH, _logger, **PGVECTOR_CREDENTIALS)

    _logger.info("Start encoding images...")
    image_ids_count = 0
    for image_ids_chunk in client.not_encoded_id_chunked_iter():
        image_ids_count += len(image_ids_chunk)
        client.add_embeddings(image_ids_chunk)

    _logger.info(f"Encoded {image_ids_count} images")

if __name__ == "__main__":
    encoder_images()
