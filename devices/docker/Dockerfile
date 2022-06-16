FROM ubuntu:18.04
SHELL ["/bin/bash", "-c"]

ARG MODEL_STORAGE_CONNECTION_STRING
ARG SNPE_SDK
ARG SNPE_SDK_FILENAME
ARG SNPE_ROOT
ARG ANDROID_NDK
ARG ANDROID_NDK_FILENAME
ARG ANDROID_NDK_ROOT

RUN apt-get update && apt-get install -y git unzip curl python3 python3-pip cmake ffmpeg libsm6 libxext6 wget locales

# have to ensure default locale is utf-8 otherwise python bails with this error:
# UnicodeEncodeError: 'ascii' codec can't encode character '\xe7' in position 17: ordinal not in range(128)
RUN locale-gen en_US.UTF-8

WORKDIR /home/archai

RUN echo "export MODEL_STORAGE_CONNECTION_STRING=\"$MODEL_STORAGE_CONNECTION_STRING\"" >> /home/archai/.profile
RUN echo "export SNPE_ROOT=$SNPE_ROOT" >> /home/archai/.profile
RUN echo "export ANDROID_NDK_ROOT=$ANDROID_NDK_ROOT" >> /home/archai/.profile
RUN echo "export INPUT_DATASET=/home/archai/datasets/FaceSynthetics" >> /home/archai/.profile
RUN echo "export PATH=$PATH:/home/archai" >> /home/archai/.profile
RUN echo "export LC_ALL=en_US.UTF-8" >> /home/archai/.profile

RUN curl -O --location "${SNPE_SDK}"
RUN unzip "${SNPE_SDK_FILENAME}"
RUN curl -O --location "${ANDROID_NDK}"
RUN unzip "${ANDROID_NDK_FILENAME}"

RUN wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1

# this echo is a trick to bypass docker build cache.
# simply change the echo string every time you want docker build to pull down new bits.
RUN echo '06/16/2022 10:16 AM' >/dev/null && git clone "https://github.com/microsoft/archai.git"

RUN source /home/archai/.profile && \
    pushd /home/archai/archai/devices && \
    python3 --version && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

COPY run.sh /home/archai/run.sh
RUN ls -al /home/archai
RUN cat run.sh

RUN chmod u+x /home/archai/run.sh

CMD ["bash", "-c", "/home/archai/run.sh"]
