import logging
import gradio as gr

from .pgvector_client import PGVectorClient
from .llm_observer import observer as llm_observer

_logger = logging.getLogger(__name__)

class Service(gr.Blocks):
    def __init__(self, images_path: str, pgvector_credentials: dict, **kwargs):
        super().__init__(**kwargs)
        self.database = PGVectorClient(images_path, [llm_observer], **pgvector_credentials)

        with self:
            gr.Markdown('## Добавляет новые данные в датасет для обучения CLIP-модели')
            self.image_url = gr.File(label='Текстовый файл со списком URL-адресов изображений')
            self.button = gr.Button('Добавить')
            self.button.click(fn=self.upload_urls, inputs=[self.image_url])

    def upload_urls(self, file_obj):
        if file_obj is None:
            return "No file uploaded."

        file_path = file_obj.name
        with open(file_path, 'r') as f:
            urls = f.readlines()
            self.database.add_images(urls)
        _logger.warning(self.database.get_summary_info())
