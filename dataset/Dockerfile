FROM python:3.10-slim

WORKDIR /app

ADD requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ADD impl /app/impl
ADD gradio_service.py /app/gradio_service.py

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "gradio_service.py"]
