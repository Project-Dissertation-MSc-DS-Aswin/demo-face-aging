FROM python:3.7

RUN apt-get update && apt-get install python3-pip wget unzip git curl -y

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

RUN apt-get install libgl1-mesa-glx -y

WORKDIR /home/project

ENV PORT 8080

COPY ./demo /home/project/demo

WORKDIR /home/project/demo

RUN wget https://project-dissertation.s3.eu-west-2.amazonaws.com/facenet_keras.h5 -P /home/project/src/models/

COPY ./src /home/project/src

CMD ["python3", "main.py"]
