FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN mkdir /home/root && useradd -ms /bin/bash user

# System Requirements
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y net-tools vim git htop wget curl zip unzip build-essential dpkg \
    iputils-ping libssl-dev libglpk-dev gdb libgoogle-glog-dev libboost-program-options-dev cmake ca-certificates clang ntpdate gnupg \
    clang-tidy clang-format lsb-release netbase valgrind tmux libsuperlu-dev libfftw3-dev libadolc-dev libmpfr-dev 


WORKDIR /workspace

# [待最终稳定后再加] 在构建镜像时清理 apt 缓存，减小最终镜像的体积
# RUN apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]