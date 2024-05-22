# 백엔드에 대한 Dockerfile
FROM python:3.10-slim

RUN mkdir /backend
COPY . /backend
WORKDIR /backend

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install gcc

RUN pip install -r requirements.txt

CMD ["python", "main.py"]