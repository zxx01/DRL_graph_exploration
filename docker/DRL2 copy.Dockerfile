# 使用支持 CUDA 的基础镜像
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# 安装系统的 Python 3.8
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    curl \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.8 get-pip.py \
    && rm get-pip.py

# 安装 PyTorch 和 torchvision
RUN pip install --no-cache-dir \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

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
    build-essential \
    && rosdep init \
    && rosdep update

# 安装其他特定依赖
RUN pip install --no-cache-dir \

# 设置工作目录
WORKDIR /workspace

# 启动命令
CMD ["bash"]