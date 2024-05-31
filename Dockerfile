#FROM python:3.10.9-slim
FROM tensorflow/tensorflow:2.11.0

COPY requirements.txt  ./
RUN apt update -y &&\
    pip install --upgrade pip &&\
    pip install -r requirements.txt
RUN pip cache remove "*"
EXPOSE 80
COPY app/* ./app/
WORKDIR /app
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]