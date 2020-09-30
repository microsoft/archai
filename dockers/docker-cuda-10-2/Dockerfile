FROM nvidia/cuda:10.2-devel

# Labels for the docker
LABEL description="Archai docker with pytorch 1.6 cuda 10.2" \
      repository="archaicuda102" \
      tag="latest" \
      creator="dedey" tooltype="pytorch" \
      tooltypeversion="1.6.0" \
      createtime="09/28/2020"

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


# Install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Create archai environment
RUN conda create -n archai python=3.7 && \
    echo "source activate archai" >> ~/.bashrc

ENV PATH /miniconda/envs/archai/bin:$PATH

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "archai", "/bin/bash", "-c"]

RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install archai
RUN git clone https://github.com/microsoft/archai.git
RUN cd archai
RUN install.sh


