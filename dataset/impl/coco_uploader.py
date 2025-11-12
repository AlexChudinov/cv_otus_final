import argparse
import logging
import os
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from .constants import DATASET, LOGGER_FORMAT, PGVECTOR_CREDENTIALS
from .pgvector_client import PGVectorClient


logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def chunked_iter(iterable, chunk_size=100):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

def init_pgvector_dataset(images_path: str, **credentials):
    client = PGVectorClient(images_path, _logger, **credentials)

    train_images_path: Path = Path(images_path) / DATASET / "train"
    _logger.info("Adding train images to the dataset...")
    image_ids = client.add_images("train", map(str, train_images_path.rglob("*.jpg")))
    _logger.info("Added %d images to the dataset", len(image_ids))
    for images_id_chunk in chunked_iter(image_ids):
        client.add_embeddings(images_id_chunk)

    test_images_path = Path(images_path) / DATASET / "validation"
    _logger.info("Adding test images to the dataset...")
    image_ids = client.add_images("test", map(str, test_images_path.rglob("*.jpg")))
    _logger.info("Added %d images to the dataset", len(image_ids))
    for images_id_chunk in chunked_iter(image_ids):
        client.add_embeddings(images_id_chunk)

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
    argparser = argparse.ArgumentParser("COCO dataset downloader")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory to save the dataset")

    args = argparser.parse_args()
    abs_path = os.path.abspath(args.output_dir)
    upload_coco(abs_path)
    init_pgvector_dataset(abs_path, **PGVECTOR_CREDENTIALS)
