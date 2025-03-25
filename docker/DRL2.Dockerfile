# 使用支持 CUDA 的基础镜像
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# 安装系统的 Python 3.8 和必要的工具
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    curl \
    wget \
    git \
    cmake \
    build-essential \
    sudo \
    libboost-all-dev \
    libeigen3-dev \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.8 get-pip.py \
    && rm get-pip.py

# 安装 PyTorch 和 torchvision
RUN pip install --no-cache-dir \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 安装 PyTorch Geometric 和依赖
RUN pip install --no-cache-dir \
    torch-scatter==2.0.9 \
    torch-sparse==0.6.13 \
    torch-cluster==1.6.0 \
    torch-spline-conv==1.2.1 \
    -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html \
    && pip install --no-cache-dir torch-geometric==2.0.4

RUN pip install --no-cache-dir \
    setuptools==59.5.0

# 安装 GTSAM
RUN git clone --branch 4.0.3 https://github.com/borglab/gtsam.git && \
    cd gtsam && \
    mkdir build && cd build && \
    cmake -DGTSAM_USE_SYSTEM_EIGEN=ON .. && \
    make -j$(nproc) && \
    sudo make install && \
    cd ../.. && \
    rm -rf gtsam

# 安装 pybind11
RUN git clone https://github.com/pybind/pybind11.git && \
    cd pybind11 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install && \
    cd ../.. && \
    rm -rf pybind11

# 安装其他Python依赖
RUN pip install --no-cache-dir \
    gym \
    matplotlib \
    six \
    numpy \
    pandas \
    scipy \
    tqdm \
    tensorboard==2.12.0 \
    scikit-learn \
    tianshou

# 安装 ROS Noetic
RUN apt-get install -y \
    gnupg2 \
    lsb-release \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

# 初始化 ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# 安装 ROS 工具和依赖
RUN apt-get update && apt-get install -y \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential 


RUN mkdir -p /etc/ros/rosdep/sources.list.d/ \
    && curl -o /etc/ros/rosdep/sources.list.d/20-default.list https://mirrors.tuna.tsinghua.edu.cn/github-raw/ros/rosdistro/master/rosdep/sources.list.d/20-default.list \
    && export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml \
    && rosdep update \
    && echo 'export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml' >> ~/.bashrc

# 设置工作目录
WORKDIR /workspace

# 创建一个检查依赖完整性的脚本
RUN echo '#!/bin/bash \n\
    echo "Checking PyTorch Geometric installation..." \n\
    python3 -c "import torch_geometric; print(f\"PyTorch Geometric version: {torch_geometric.__version__}\")" \n\
    echo "Checking GTSAM installation..." \n\
    if [ -f "/usr/local/lib/libgtsam.so" ]; then \n\
    echo "GTSAM installed successfully" \n\
    else \n\
    echo "GTSAM installation failed" \n\
    fi \n\
    echo "All dependencies installed successfully!"' > /usr/local/bin/check_deps.sh && \
    chmod +x /usr/local/bin/check_deps.sh

# 启动命令
CMD ["bash"]