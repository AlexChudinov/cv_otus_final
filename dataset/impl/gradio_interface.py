import logging
import gradio as gr

from .pgvector_client import PGVectorClient

_logger = logging.getLogger(__name__)

class Service(gr.Blocks):
    def __init__(self, images_path: str, pgvector_credentials: dict, **kwargs):
        super().__init__(**kwargs)
        self.database = PGVectorClient(images_path, **pgvector_credentials)

        with self:
            gr.Markdown('## Добавляет новые данные в датасет для обучения CLIP-модели')
            self.image_url = gr.Text(label='URL изображения')
            self.button = gr.Button('Найти')
