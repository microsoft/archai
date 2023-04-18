FROM ubuntu:18.04
SHELL ["/bin/bash", "-c"]

ARG SNPE_SDK_ZIP
ARG SNPE_SDK_ROOT
ARG ANDROID_NDK_ZIP
ARG ANDROID_NDK_ROOT

RUN apt-get update && apt-get install -y build-essential libtool autoconf cmake unzip wget git unzip curl python3 python3-dev python3-distutils python3-pip ffmpeg libsm6 libxext6 wget locales libjpeg-dev zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler

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
    git checkout v3.19.4 && \
    git submodule update --init --recursive && \
    mkdir build_source && cd build_source && \
    cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# have to ensure default locale is utf-8 otherwise python bails with this error:
# UnicodeEncodeError: 'ascii' codec can't encode character '\xe7' in position 17: ordinal not in range(128)
RUN locale-gen en_US.UTF-8

# copy and unzip the SNPE SDK
WORKDIR /home/archai/snpe
COPY "${SNPE_SDK_ZIP}" .
RUN unzip "${SNPE_SDK_ZIP}"

# Copy and unzip the Android NDK
WORKDIR /home/archai/ndk
COPY "${ANDROID_NDK_ZIP}" .
RUN unzip "${ANDROID_NDK_ZIP}"

WORKDIR /home/archai

ENV AZUREML_CONDA_ENVIRONMENT_PATH="/home/archai/miniconda3/snap37"
ENV SNPE_ROOT="/home/archai/snpe/$SNPE_SDK_ROOT"
ENV SNPE_ANDROID_ROOT="/home/archai/snpe/$SNPE_SDK_ROOT"
ENV ANDROID_NDK_ROOT="/home/archai/ndk/$ANDROID_NDK_ROOT"
ENV PATH="/home/archai/miniconda3/bin:$PATH:/home/archai:/home/archai/ndk/tools/${PLATFORM_TOOLS_ROOT}:/home/archai/snpe/${SNPE_SDK_ROOT}/bin/x86_64-linux-clang"
ENV LC_ALL="en_US.UTF-8"
ENV PYTHONPATH="/home/archai/snpe/$SNPE_SDK_ROOT/lib/python"
ENV LD_LIBRARY_PATH="${AZUREML_CONDA_ENVIRONMENT_PATH}/lib:${SNPE_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda3
RUN conda init bash
RUN conda create -y -n snap37 python=3.8 pip=20.2.4
ENV PATH="${AZUREML_CONDA_ENVIRONMENT_PATH}/bin:$PATH"

RUN wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1

# this echo is a trick to bypass docker build cache.
# simply change the echo string every time you want docker build to pull down new bits.
RUN echo '04/18/2023 12:22 PM' >/dev/null && git clone https://github.com/microsoft/archai.git
RUN cd archai && git checkout clovett/aml && pip install -e .[dev]

RUN echo "using this pip version: " && which pip
RUN echo "using this python version: " && which python

RUN pushd /home/archai/archai/tasks/face_segmentation/aml && \
    python --version && \
    pip install -r requirements.txt

RUN pip list

# This container starts running immediately so that the kubernetes cluster simply scales
# automatically based on how busy this script is (see HorizontalPodAutoscaler in quantizer.yaml).
COPY run.sh /home/archai/run.sh
RUN ls -al /home/archai
RUN cat run.sh

RUN chmod u+x /home/archai/run.sh
CMD ["bash", "-c", "/home/archai/run.sh"]
