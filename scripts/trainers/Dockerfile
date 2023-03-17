# Defines the base image
FROM nvcr.io/nvidia/pytorch:23.02-py3

# Installs desired packages for the workload
RUN apt-get update && \
    apt-get install --no-install-recommends --no-install-suggests -yq && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean

# DeepSpeed and OpenMPI
RUN pip install --upgrade pip && \
    pip uninstall -y xgboost && \
    DS_BUILD_UTILS=1 DS_BUILD_FUSED_LAMB=1 pip install deepspeed==0.8.1 && \
    CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

# Flash-Attention and CUDA extensions for cross-entropy, fused dense, layer norm
RUN pip install flash-attn==0.2.8
RUN git clone https://github.com/HazyResearch/flash-attention \
    && cd flash-attention && git checkout v0.2.8 \
    && cd csrc/fused_softmax && pip install . && cd ../../ \
    && cd csrc/rotary && pip install . && cd ../../ \
    && cd csrc/xentropy && pip install . && cd ../../ \
    && cd csrc/layer_norm && pip install . && cd ../../ \
    && cd csrc/fused_dense_lib && pip install . && cd ../../ \
    # && cd csrc/ft_attention && pip install . && cd ../../ \
    && cd .. && rm -rf flash-attention

# Archai (development)
RUN pip install --user git+https://github.com/microsoft/archai.git#egg=archai[dev]