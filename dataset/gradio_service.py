import os
from impl.gradio_interface import Service

if __name__ == '__main__':
    pgvector_credentials = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": 5432,
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DB"),
    }

    service = Service(os.getenv("IMAGES_PATH"), pgvector_credentials)
    service.launch()
