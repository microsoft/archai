# CodeGen + DeepSpeed

This folder contains the necessary files and instructions to train a CodeGen model using DeepSpeed and Flash-Attention.

## Installation

To support DeepSpeed and Flash-Attention, you need to install Archai. You can do so by running the following command in your terminal:

```bash
pip install --user git+https://github.com/microsoft/archai.git#egg=archai[dev,deepspeed,flash-attn]
```

*Please note that DeepSpeed is not compatible with Windows.*

### Docker

Alternatively, you can use the following Docker-based configuration to build a Docker image with Archai and all the necessary dependencies:

```Dockerfile
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
```

## Data Preparation

To prepare the data, you can use the `FastHfDatasetProvider` class to load and encode datasets from the Hugging Face Hub. Here is an example code:

```Python
dataset_provider = FastHfDatasetProvider.from_hub(
    "wikitext",
    dataset_config_name="wikitext-103-raw-v1",
    tokenizer_name="Salesforce/codegen-350M-mono",
    cache_dir="wikitext_cache",
)
train_dataset = dataset_provider.get_train_dataset(seq_len=2048)
eval_dataset = dataset_provider.get_val_dataset(seq_len=2048)
```

Using `FastHfDatasetProvider` is recommended as it offers a faster way to load and encode datasets. Once the dataset is encoded, it can be cached and loaded from disk later as follows:

```Python
dataset_provider = FastHfDatasetProvider.cache("wikitext_cache")
```

## Training

To begin the training, run the following command:

```bash
deepspeed deepspeed_train_codegen.py -ds ds_config.json
```

You can customize the training by modifying the arguments defined in `CodeGenFlashConfig`, `DsTrainingArguments`, and `ds_config.json`. By default, the arguments are set to perform a toy training and explain how the pipeline works.
