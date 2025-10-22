import uuid
import os
import requests

from io import BytesIO
from pathlib import Path
from PIL import Image

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

class Dataset:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def _update_dvc_caches(self):
       os.system('dvc add -R ' + str(self.data_dir))
       os.system('dvc push')

    def add_data(self, image_url: str, prompt: str):
        new_uuid = str(uuid.uuid4())
        image_name = new_uuid + '.jpeg'
        text_prompt = new_uuid + '.txt'
        download_image(image_url).save(self.data_dir / image_name)
        with open(self.data_dir / text_prompt, 'w') as f:
            f.write(prompt)
