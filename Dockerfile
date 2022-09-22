FROM ubuntu:20.04

RUN apt-get update && apt-get install python3 python3-pip wget unzip git curl -y

WORKDIR /home/project

ARG DEBIAN_FRONTEND=noninteractive
RUN set -ex && apt-get install build-essential python3-distutils python3-apt -y

RUN apt install software-properties-common -y
RUN wget https://bootstrap.pypa.io/get-pip.py -P /home/project
RUN python3 get-pip.py
COPY ./requirements.txt /home/project
RUN apt-get install python3-dev -y
RUN pip3 install -r requirements.txt

RUN pip3 install fastapi uvicorn pydantic python-socketio eventlet 
RUN pip3 install opencv-python
RUN pip3 install matplotlib

COPY src/models /home/project/src/models

WORKDIR /home/project

ENV PORT 8080

WORKDIR /home/project/demo

CMD ["python", "main.py"]
