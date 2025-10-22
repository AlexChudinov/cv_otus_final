# Сервис для добавления новых данных в датасет. Датасет представляет собой dvc repo, в котором хранятся изображения с описывающим их промптом

import gradio as gr

service = gr.Interface()


if __name__ == '__main__':
    service.launch(host='0.0.0.0')
