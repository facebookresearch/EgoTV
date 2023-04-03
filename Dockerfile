FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04 
ENV DEBIAN_FRONTEND=nonintercative
RUN apt-get update \
    && \
    echo "------------------------------------------------------ essentials" \
    && \
    apt-get install -y --no-install-recommends -y \
    build-essential \
    apt-utils \
    python3-setuptools \
    git-all \
    openjdk-8-jdk \
    && \
    echo "------------------------------------------------------ editors" \
    && \
    apt-get install -y --no-install-recommends -y \
    emacs \
    vim \
    nano \
    && \
    echo "------------------------------------------------------ software" \
    && \
    apt-get install -y --no-install-recommends -y \
    python3-pip \
    tmux \
    screen \
    graphviz \
    cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig \
    sumo \
    sumo-tools \
    sumo-doc \
    && \
    echo "------------------------------------------------------ cleanup" \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# RUN brew install swig
RUN pip3 install --upgrade pip
COPY docker_requirements.txt .
RUN pip3 install -r docker_requirements.txt
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -c "import nltk; nltk.download('punkt')"
