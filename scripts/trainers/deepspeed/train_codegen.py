# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os

import torch
from transformers import AutoTokenizer

from archai.datasets.nlp.fast_hf_dataset_provider import (
    FastDataCollatorForLanguageModeling,
    FastHfDatasetProvider,
)
from archai.discrete_search.search_spaces.nlp.tfpp.modeling_codegen_flash import (
    CodeGenFlashConfig,
    CodeGenFlashSequential,
)
from archai.trainers.nlp.ds_trainer import DsTrainer
from archai.trainers.nlp.ds_training_args import DsTrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains/Fine-tunes a CodeGen model with DeepSpeed.")

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-dcn",
        "--dataset_config_name",
        type=str,
        default="wikitext-103-raw-v1",
        help="Configuration name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "-ptm",
        "--pre_trained_model_path",
        type=str,
        default=None,
        help="Path to the pre-trained model.",
    )

    parser.add_argument(
        "-ds",
        "--ds_config_path",
        type=str,
        default=None,
        help="Path to the DeepSpeed configuration file.",
    )

    parser.add_argument(
        "-pps",
        "--pipe_parallel_size",
        type=int,
        default=1,
        help="Size of pipeline parallelism.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Defines an output folder for the saved outputs.",
    )

    parser.add_argument(
        "-l",
        "--local_rank",
        type=int,
        default=-1,
        help="Rank of process passed by the DeepSpeed launcher.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    collator = FastDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset_provider = FastHfDatasetProvider.from_hub(
        args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        tokenizer=tokenizer,
    )
    train_dataset = dataset_provider.get_train_dataset(seq_len=2048)
    eval_dataset = dataset_provider.get_val_dataset(seq_len=2048)

    config = CodeGenFlashConfig(
        vocab_size=50304,
        n_positions=2048,
        n_embd=1024,
        n_layer=20,
        n_head=16,
        rotary_dim=32,
        pad_vocab_size_multiple=64,
        attn_type="flash",
        use_fused_mlp=True,
    )
    model = CodeGenFlashSequential(config)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    if args.pre_trained_model_path is not None:
        state_dict = torch.load(os.path.join(args.pre_trained_model_path, "mp_rank_00_model_states.pt"))
        if state_dict["module"] is not None:
            model.load_state_dict(state_dict["module"])
        else:
            for i, layer in enumerate(model.layers):
                state_dict = torch.load(os.path.join(args.pre_trained_model_path, f"layer_{i:02d}-model_states.pt"))
                layer.load_state_dict(state_dict)

    training_args = DsTrainingArguments(
        "ds-codegen",
        ds_config=args.ds_config_path,
        local_rank=args.local_rank,
        max_steps=1000,
        logging_steps=10,
        save_steps=1000,
        eval_steps=250,
        eval_max_steps=25,
        pipe_parallel_size=args.pipe_parallel_size,
        pipe_parallel_loss_fn=model.loss,
    )
    trainer = DsTrainer(
        model=model.layers if args.pipe_parallel_size > 0 else model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
