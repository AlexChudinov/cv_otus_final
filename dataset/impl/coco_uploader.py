import argparse
import logging
import os

import fiftyone as fo
import fiftyone.zoo as foz

from .constants import LOGGER_FORMAT


logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
_logger = logging.getLogger(__name__)


def upload_coco(output_dir: str) -> None:
    fo.config.dataset_zoo_dir = output_dir
    _logger.info("Downloading COCO dataset to directory %s...", output_dir)

    train_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        dataset_name="coco-2017-train-detections" # Optional: assign a custom name
    )
    _logger.info("COCO train dataset info:")
    _logger.info(train_dataset)

    val_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        dataset_name="coco-2017-val-detections" # Optional: assign a custom name
    )
    _logger.info("COCO validation Dataset info:")
    _logger.info(val_dataset)


def run_fiftyone_app(output_dir: str) -> None:
    fo.config.dataset_zoo_dir = output_dir
    _logger.info("Starting FiftyOne app with COCO dataset in directory %s...", output_dir)
    session = fo.launch_app(address="0.0.0.0")
    session.wait(-1)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("COCO dataset downloader")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory to save the dataset")

    args = argparser.parse_args()
    abs_path = os.path.abspath(args.output_dir)
    upload_coco(abs_path)
    run_fiftyone_app(abs_path)
