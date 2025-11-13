import os

DATASET = "coco-2017"

LOGGER_FORMAT='[%(asctime)s] %(levelname)s: %(message)s'

PGVECTOR_CREDENTIALS = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": 5432,
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "database": os.getenv("POSTGRES_DB"),
}

IMAGES_PATH  = os.getenv("IMAGES_PATH")
