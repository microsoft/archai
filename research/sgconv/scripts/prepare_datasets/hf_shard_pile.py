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
    num_shards = 21
    orig_dataset = load_from_disk("/data/pile_smraiz_10")

    for i in tqdm(range(num_shards)):
        dataset = orig_dataset.shard(num_shards, index=i)  # shard is problematic
        print(f"saving sharded dataset of length {dataset.num_rows}, index {i}")

        # save shard to disk
        savename = os.path.join("/data/pile_smraize_10_sharded", f"pile_index_{i:04d}_of_{num_shards}")
        dataset.save_to_disk(savename)

        print("done.")


if __name__ == "__main__":
    main()
