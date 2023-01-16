# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Root image to be based
# Available images: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.10-py3

# Labels for the docker
LABEL description="NVIDIA Docker with Archai" \
      repository="archai" \
      tag="latest" \
      creator="microsoft" \
      tooltype="archai" \
      tooltypeversion="1.0.0" \
      createtime="11/08/2022"

# Exports environment variables
ENV PATH="/root/.local/bin:$PATH"

# Installs basic utilities
RUN set -e -o xtrace
RUN apt-get update && apt-get install -y
RUN apt-get install -y apt-utils git

# Installs Archai
# Note that APEX is not needed because it comes with NVIDIA's image
RUN git clone -b v1.0.0 --single-branch https://github.com/microsoft/archai.git
WORKDIR /workspace/archai
RUN pip install --user --no-cache-dir .[dev]
