FROM nvcr.io/nvidia/pytorch:21.09-py3

# Labels for the docker
LABEL description="Archai docker with all dependencies" \
      repository="archai" \
      tag="latest" \
      creator="shitals" tooltype="pytorch" \
      tooltypeversion="0.5.0" \
      createtime="10/8/2021"

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip install --no-cache-dir git+https://github.com/microsoft/archai.git