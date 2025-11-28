from .constants import IMAGES_PATH, PGVECTOR_CREDENTIALS
from .pgvector_client import PGVectorClient

from .utils import create_logger

_logger = create_logger()

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
