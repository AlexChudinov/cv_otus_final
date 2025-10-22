# Сервис для добавления новых данных в датасет. Датасет представляет собой dvc repo, в котором хранятся изображения с описывающим их промптом

from impl.interface import Service

if __name__ == '__main__':
    service = Service("./datafiles")
    service.launch()
