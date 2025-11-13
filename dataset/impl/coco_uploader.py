import logging
import os
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from .constants import DATASET, IMAGES_PATH, LOGGER_FORMAT, PGVECTOR_CREDENTIALS
from .pgvector_client import PGVectorClient
from .utils import chunked_iter

logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def init_pgvector_dataset(images_path: str, **credentials):
    client = PGVectorClient(images_path, _logger, **credentials)

    train_images_path: Path = Path(images_path) / DATASET / "train"
    _logger.info("Adding train images to the dataset...")

    file_count = 0
    file_uploaded_count = 0
    for filename in chunked_iter(train_images_path.rglob("*.jpg")):
        file_count += len(filename)
        filename_not_in_db = client.filter_file_not_exist(filename)
        image_ids = client.add_images("train", filename_not_in_db)
        file_uploaded_count += len(image_ids)

    _logger.info("Files count in train dataset: %d", file_count)
    _logger.info("Added %d images to train dataset", file_uploaded_count)
    # for images_id_chunk in chunked_iter(image_ids):
    #     client.add_embeddings(images_id_chunk)

    test_images_path = Path(images_path) / DATASET / "validation"
    _logger.info("Adding test images to the dataset...")

    file_count = 0
    file_uploaded_count = 0
    for filename in chunked_iter(test_images_path.rglob("*.jpg")):
        file_count += len(filename)
        filename_not_in_db = client.filter_file_not_exist(filename)
        image_ids = client.add_images("validation", filename_not_in_db)
        file_uploaded_count += len(image_ids)

    _logger.info("Files count in validation dataset: %d", file_count)
    _logger.info("Added %d images to validation dataset", file_uploaded_count)
    # for images_id_chunk in chunked_iter(image_ids):
    #     client.add_embeddings(images_id_chunk)

def upload_coco(output_dir: str) -> None:
    fo.config.dataset_zoo_dir = output_dir
    _logger.info("Downloading COCO dataset to directory %s...", output_dir)

    if not os.path.exists(os.path.join(output_dir, "coco-2017", "train")):
        train_dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            dataset_name="coco-2017-train-detections" # Optional: assign a custom name
        )
        _logger.info("COCO train dataset info:")
        _logger.info(train_dataset)

    if not os.path.exists(os.path.join(output_dir, "coco-2017", "validation")):
        val_dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            label_types=["detections"],
            dataset_name="coco-2017-val-detections" # Optional: assign a custom name
        )
        _logger.info("COCO validation Dataset info:")
        _logger.info(val_dataset)


if __name__ == "__main__":
    abs_path = os.path.abspath(IMAGES_PATH)
    upload_coco(abs_path)
    init_pgvector_dataset(abs_path, **PGVECTOR_CREDENTIALS)
