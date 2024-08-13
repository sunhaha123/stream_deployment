# 使用PyTorch基础镜像
FROM docker-register-registry-vpc.cn-shanghai.cr.aliyuncs.com/well-known/pytorch:2.1.2-cuda12.1-cudnn8-devel as builder


# 设置环境变量以避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 设置时区
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

# 设置代理
ENV HTTP_PROXY=10.18.252.1:6152 HTTPS_PROXY=10.18.252.1:6152
RUN echo "代理设置完成: $HTTP_PROXY"

# 复制项目文件
COPY . /opt/removebg_deployment
RUN echo "项目文件已复制到 /opt/removebg_deployment"

# 设置工作目录
WORKDIR /opt/removebg_deployment
RUN echo "当前工作目录: $(pwd)"


# 更新并安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    zlib1g-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置pip源
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN echo "使用的pip源: $PIP_INDEX"

# 安装依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --index-url ${PIP_INDEX} && \
    echo "依赖安装完成"

# 清除代理设置
ENV HTTP_PROXY="" HTTPS_PROXY=""
RUN echo "代理设置已清除"

# 列出安装的文件
RUN echo "项目文件列表:" && ls -R /opt/removebg_deployment

# 打印Python版本和pip列表
RUN echo "Python版本:" && python --version && \
    echo "已安装的pip包:" && pip list



# 设置默认命令（可选）
# CMD ["python", "/opt/removebg_deployment/main/app_remove_multi.py"]