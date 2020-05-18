FROM nvidia/cuda:10.1-devel

# Labels for the docker
LABEL description="This docker has the environment for running petridish experiments in pytorch" \
      repository="archaicuda101" \
      tag="latest" \
      creator="dedey" tooltype="pytorch" \
      tooltypeversion="1.3.0" \
      createtime="10/17/2019"

RUN apt-get update -y && \
    apt-get -y install \
        gcc \
        g++ \
        curl \
        ca-certificates \
        bzip2 \
        cmake \
        tree \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
	    swig \
        cmake \
        build-essential \
        zlib1g-dev \
        libosmesa6-dev \
        python-pygame \
        python-scipy \
        patchelf \
        libglfw3-dev \ 
        git \
	    libglew-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /miniconda.sh && \
    /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

ADD archaicuda101.yml /tmp/archaicuda101.yml

RUN conda env create -f /tmp/archaicuda101.yml && \
    echo "source activate archaicuda101" >> ~/.bashrc

ENV PATH /miniconda/envs/archaicuda101/bin:$PATH

ADD requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
