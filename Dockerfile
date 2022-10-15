FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /usr/src/project

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD sh -c "python3 /usr/src/project/yolov5/main.py"
