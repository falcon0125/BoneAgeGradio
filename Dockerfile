FROM python:3.9.19-alpine
COPY requirements.txt  ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip cache remove "*"
EXPOSE 80
COPY app/* ./app/
WORKDIR /app
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app-tflite.py"]