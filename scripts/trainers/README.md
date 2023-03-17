# Training Models with Archai

This folder contains the necessary files and instructions to train models using Archai.

## Installation

Before you can start training models, you need to install Archai. To do so, you can follow these instructions:

1. Open your terminal and run the following command:

    ```bash
    pip install --user git+https://github.com/microsoft/archai.git#egg=archai[nlp]
    ```

2. If you plan to use DeepSpeed and Flash-Attention, run this command instead:

    ```bash
    pip install --user git+https://github.com/microsoft/archai.git#egg=archai[nlp,deepspeed,flash-attn]
    ```

*Please note that DeepSpeed is not compatible with Windows.*

Alternatively, you can use Docker to build a Docker image with Archai and all the necessary dependencies. Simply follow the instructions in the `Dockerfile` provided in this folder.

## Data Preparation

To prepare the data, you can use the `FastHfDatasetProvider` class to load and encode datasets from the Hugging Face Hub. This is recommended as it offers a faster way to load and encode datasets. Here is an example code:

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

Once the dataset is encoded, it can be cached and loaded from disk later as follows:

```Python
dataset_provider = FastHfDatasetProvider.cache("wikitext_cache")
```

However, please note that this method does not apply for NVIDIA-related training, as datasets are automatically created and encoded.

## DeepSpeed

If you are using DeepSpeed, run the following command to begin training:

```bash
deepspeed deepspeed/train_codegen.py --help
```

You can customize the training by modifying the arguments defined in `CodeGenFlashConfig`, `DsTrainingArguments`, and `ds_config.json`. By default, the arguments are set to perform a toy training and explain how the pipeline works.

## Hugging Face

If you are using Hugging Face, run the following command to begin training:

```bash
python -m torch.distributed.run --nproc_per_node=4 hf/train_codegen.py --help
```

You can customize the training by modifying the arguments defined in `CodeGenConfig` and `TrainingArguments`. By default, the arguments are set to perform a toy training and explain how the pipeline works.

## NVIDIA

If you are using NVIDIA, run the following command to begin training:

```bash
python -m torch.distributed.run --nproc_per_node=4 nvidia/train_gpt2.py --help
```

You can customize the training by modifying the arguments defined in `GPT2Config` and `NvidiaTrainingArguments`. By default, the arguments are set to perform a toy training and explain how the pipeline works.
