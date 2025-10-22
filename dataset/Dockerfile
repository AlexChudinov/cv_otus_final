FROM python:3.13-slim

WORKDIR /app

ADD gradio_service.py /app/gradio_service.py
ADD requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

CMD ["python", "gradio_service.py"]
