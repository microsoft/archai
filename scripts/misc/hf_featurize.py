"""
Script for featurizing a large dataset like The Pile, with the 
motivation to use downstream data summarization techniques on it.
"""
import os
import argparse

import torch
import numpy as np

from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default= "facebook/opt-350m", 
                        help="Name of HF checkpoint to use as featurizer", 
                        type=str)
    parser.add_argument("--max_length", default=512,
                        help="Max length of input sequence to process.",
                        type=int)
    parser.add_argument("--savedir", default="/home/dedey/smraiz/input/",
                        help="Save directory for storing features",
                        type=str)
    parser.add_argument("--num-shards", default=2100,
                        help="Number of shards to create. WARNING: Don't change across runs on the same dataset.",
                        type=int)
    parser.add_argument("--shard-index", default=0,
                        help="Particular shard to process. Must be less than num_shards",
                        type=int)
    parser.add_argument("--batch-size", default=42,
                        help="Batch size for feature extraction.",
                        type=int)
    parser.add_argument("--gpuid", default=0,
                        help="GPU id to use. -1 will use cpu.",
                        type=int)
    args = parser.parse_args()

    # set device
    if torch.cuda.is_available() and args.gpuid >= 0:
        device = f"cuda:{args.gpuid}"
    else:
        device = "cpu"

    checkpoint = args.checkpoint
    model = AutoModel.from_pretrained(checkpoint)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # load the pile train set
    dataset = load_dataset("the_pile", split='train')

    # store all features in a list
    f_container = []

    # function to featurize a batch of examples
    def featurize(examples):
        encoded_input = tokenizer(examples["text"], 
                        return_tensors='pt', 
                        padding=True,
                        truncation=True,
                        max_length=args.max_length).to(device)
        output = model(**encoded_input)
        fs = output['last_hidden_state'].detach().cpu().numpy()
        fs = np.reshape(fs, (fs.shape[0], -1))
        f_container.append(fs)

        # can return this to create a new column
        # in the dataset itself if required
        # return {"embedding" : output['last_hidden_state'].detach().cpu().numpy()}

    dataset = dataset.shard(args.num_shards, index=args.shard_index)
    print(f"Sharded dataset length: {dataset.num_rows}")
    # num_proc > 1 will fail on GPU
    _ = dataset.map(featurize, 
                    batched=True, 
                    batch_size=args.batch_size, 
                    num_proc=1)

    features = np.concatenate(f_container, axis=0)
    savename = os.path.join(args.savedir, f"shard_index_{args.shard_index}_total_shards_{args.num_shards}_features.npy")
    print(f"saving features to {savename} with shape {features.shape}")
    np.save(savename, features)

    print('done.')

if __name__ == '__main__':
    main()