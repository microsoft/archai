"""
Script for featurizing a large dataset like The Pile, with the
motivation to use downstream data summarization techniques on it.
"""
import argparse
import os

import numpy as np
import plotly.graph_objects as go
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset


def main():

    dataset = load_from_disk("/data/codedatasets/only_python_nocomments")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(100000))
    print(f"using dataset of length {dataset.num_rows}")

    # get number of tokens in each sentence
    checkpoint = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tok_len_storage = []

    def tokenize_len(examples):
        encoded_input = tokenizer(examples["codenocomments"], return_tensors="np", padding=False, truncation=False)
        num_toks = encoded_input["input_ids"].shape[1]
        tok_len_storage.append(num_toks)

    _ = dataset.map(tokenize_len, batched=False, batch_size=1, num_proc=1)

    fig = go.Figure(data=[go.Histogram(x=tok_len_storage, histnorm="probability")])
    fig.write_html("/home/dedey/only_python_nocomments_tok_len_hist.html")
    fig.show()

    print("done.")


if __name__ == "__main__":
    main()
