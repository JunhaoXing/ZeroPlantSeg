FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


RUN useradd -u 1153 juxing

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0" 
# install python packages
RUN apt-get update && apt-get install -y python3 python3-pip 

#install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext-dev \
    libgl1-mesa-dev \
    screen \
    ninja-build

# install requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install 'git+https://github.com/facebookresearch/segment-anything.git' && pip3 install 'git+https://github.com/facebookresearch/detectron2.git' && pip3 install 'git+https://github.com/IDEA-Research/GroundingDINO.git'

WORKDIR /home/juxing/mnt/workspace/zeroshot_hierarchical