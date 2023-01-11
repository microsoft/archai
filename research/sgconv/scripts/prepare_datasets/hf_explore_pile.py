"""
Script for featurizing a large dataset like The Pile, with the
motivation to use downstream data summarization techniques on it.
"""
import argparse
import os

import numpy as np
import plotly.graph_objects as go
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset


def main():

    dataset = load_dataset("the_pile", split="train")
    dataset = dataset.shard(21000, index=1)
    print(f"using dataset of length {dataset.num_rows}")

    # get length of words in each sentence
    # len_storage = []
    # for dat in tqdm(dataset):
    #     len_storage.append(len(dat['text'].split()))

    # fig = go.Figure(data=[go.Histogram(x=len_storage, histnorm='probability')])
    # fig.write_html('pile_text_len_hist.html')
    # fig.show()

    # get number of tokens in each sentence
    checkpoint = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tok_len_storage = []

    def tokenize_len(examples):
        encoded_input = tokenizer(examples["text"], return_tensors="np", padding=False, truncation=False)
        num_toks = encoded_input["input_ids"].shape[1]
        if num_toks < 500:
            tok_len_storage.append(num_toks)

    _ = dataset.map(tokenize_len, batched=False, batch_size=1, num_proc=1)

    fig = go.Figure(data=[go.Histogram(x=tok_len_storage, histnorm="probability")])
    fig.write_html("pile_tok_len_below_500_hist.html")
    fig.show()

    print("done.")


if __name__ == "__main__":
    main()
