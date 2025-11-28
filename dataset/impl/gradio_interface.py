import gradio as gr

from .constants import IMAGES_PATH, PGVECTOR_CREDENTIALS
from .pgvector_client import PGVectorClient
from .utils import create_logger


class Service(gr.Blocks):
    def __init__(self, images_path: str, pgvector_credentials: dict, **kwargs):
        super().__init__(**kwargs)
        self.logger = create_logger()
        self.database = PGVectorClient(images_path, self.logger, **pgvector_credentials)
        self.build_layout()

    def build_layout(self):
        with self:
            self.image_url = gr.Textbox(label='URL изображения')
            self.number_of_images = gr.Slider(1, 20, value=1, label='Количество изображений', step=1, precision=1)
            self.similarity_type = gr.Radio(['cosine', 'dot', 'L2', 'L1'], value='cosine', label='Тип сходства')
            self.button = gr.Button('Найти')
            output = [gr.Image(visible=False) for _ in range(21)]
            gr.Interface(fn=self.query,
                         inputs=[self.image_url, self.number_of_images, self.similarity_type],
                         outputs=output,
                         title='Поиск похожих изображений')

    def query(self, image_url: str, number_of_images: int, similarity: str) -> list[gr.Image]:
        output, orig_image = self.database.get_neighbors(image_url, number_of_images, similarity)
        show = [
            gr.Image(value=image, visible=True, label=f'{round(similarity, 2)}')
            for similarity, image in output
        ]
        hide = [
            gr.Image(visible=False)
            for _ in range(20 - number_of_images)
        ]
        return [gr.Image(value=orig_image, visible=True, label="Запрос")] + show + hide

if __name__ == '__main__':
    service = Service(IMAGES_PATH, PGVECTOR_CREDENTIALS)
    service.launch()
