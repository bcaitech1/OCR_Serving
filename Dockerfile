# nvidia cuda
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu16.04
# pyhon & pip3
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install  -y python3.7 \
    && update-alternatives --install /usr/bin/python3 python /usr/bin/python3.7 0 \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip \
    && pip3 --version
# set work dir
WORKDIR /unnamed_OCR
RUN apt-get update \
        && apt-get install -y \
            wget \
            curl \
            && rm -rf /var/lib/apt/lists/*
COPY . ./
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN pip3 install -r requirements.txt
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
# ENV MODEL_PATH /checkpoint/model.pth ??
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
EXPOSE 5000
