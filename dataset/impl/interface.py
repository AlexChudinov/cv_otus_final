import gradio as gr

from .dataset import Dataset

class Service(gr.Blocks):
    def __init__(self, dataset_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.dataset = Dataset(dataset_dir)

        with self:
            gr.Markdown('## Добавляет новые данные в датасет для обучения CLIP-модели')
            self.prompt = gr.Textbox(label='Промпт')
            self.image_url = gr.Textbox(label='URL изображения')
            self.button = gr.Button('Добавить')
            self.button.click(fn=self.dataset.add_data, inputs=[self.image_url, self.prompt], outputs=[])
