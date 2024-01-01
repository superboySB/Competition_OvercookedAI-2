FROM nvcr.io/nvidia/pytorch:23.04-py3

# System Requirements
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev \
    libxi-dev libgl1-mesa-glx htop

# Conda
RUN mkdir ~/miniconda3 && cd ~ && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && rm miniconda3/miniconda.sh && \
    miniconda3/bin/conda init bash