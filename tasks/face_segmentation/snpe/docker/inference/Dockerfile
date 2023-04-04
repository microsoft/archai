# Qualcomm SNPE SDK requires ubuntu:18.04 and python 3.6!
FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest

ARG SNPE_SDK_ZIP
ARG SNPE_SDK_ROOT
ARG ANDROID_NDK_ZIP
ARG ANDROID_NDK_ROOT

ENV AZUREML_CONDA_ENVIRONMENT_PATH="/opt/miniconda/envs/snap"

RUN apt-get update && apt-get install -y build-essential libtool autoconf cmake unzip wget git curl python3 python3-dev python3-distutils python3-pip ffmpeg libsm6 libxext6 wget locales libjpeg-dev zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler

# need cmake 3.22 to build latest version of onnx-simplifier on Ubuntu 18
RUN apt update && \
    apt install -y software-properties-common lsb-release && \
    apt clean all
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4
RUN apt update && apt install -y cmake

# build latest version of protobuf needed by onnx
RUN git clone https://github.com/protocolbuffers/protobuf.git
RUN cd protobuf && \
    git checkout v3.16.0 && \
    git submodule update --init --recursive && \
    mkdir build_source && cd build_source && \
    cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# have to ensure default locale is utf-8 otherwise python bails with this error:
# UnicodeEncodeError: 'ascii' codec can't encode character '\xe7' in position 17: ordinal not in range(128)
RUN locale-gen en_US.UTF-8

WORKDIR /home/archai/snpe
COPY "${SNPE_SDK_ZIP}" .
RUN unzip "${SNPE_SDK_ZIP}"

WORKDIR /home/archai/ndk
COPY "${ANDROID_NDK_ZIP}" .
RUN unzip "${ANDROID_NDK_ZIP}"

WORKDIR /home/archai
copy requirements.txt .

ENV SNPE_ROOT="/home/archai/snpe/$SNPE_SDK_ROOT"
ENV ANDROID_NDK_ROOT="/home/archai/ndk/$ANDROID_NDK_ROOT"
ENV PATH="$PATH:/home/archai:/home/archai/ndk/tools/${PLATFORM_TOOLS_ROOT}:/home/archai/snpe/${SNPE_SDK_ROOT}/bin/x86_64-linux-clang"
ENV LC_ALL="en_US.UTF-8"
ENV PYTHONPATH="/home/archai/snpe/$SNPE_SDK_ROOT/lib/python"
ENV LD_LIBRARY_PATH="${AZUREML_CONDA_ENVIRONMENT_PATH}/lib:${SNPE_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}"

# initialize conda environment. (conda comes with mcr.microsoft.com/azureml)
RUN conda init bash

# this is how you activate a conda environment in Dockerfile
ENV PATH="${AZUREML_CONDA_ENVIRONMENT_PATH}/bin:$PATH"

# Create conda environment
RUN conda create -y -n snap python=3.6 pip=20.2.4

RUN pip install -r requirements.txt