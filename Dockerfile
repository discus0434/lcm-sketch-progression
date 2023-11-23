FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 CUDA_HOME=/usr/local/cuda-11.8
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-11.8" >> /etc/bash.bashrc

RUN apt-get update && apt-get install -y --no-install-recommends \
        make \
        wget \
        tar \
        build-essential \
        libgl1-mesa-dev \
        curl \
        unzip \
        git \
        python3-dev \
        python3-pip \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /lcm-sketch-progression

COPY requirements.lock pyproject.toml README.md src /lcm-sketch-progression/
RUN pip3 install -r requirements.lock

WORKDIR /app
