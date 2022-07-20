"""
Script for creating a toy example with clear duplicates
to test downstream summarization techniques on it.
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
    parser.add_argument("--savedir", default="/home/dedey/smraiz/input/",
                        help="Save directory for storing features",
                        type=str)
    
    args = parser.parse_args()

    # set device
    device = "cpu"

    checkpoint = args.checkpoint
    model = AutoModel.from_pretrained(checkpoint)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    input_1 = ['Feature extraction pipeline using no model head. This pipeline extracts the hidden states from the base transformer, which can be used as features in downstream tasks.']
    input_2 = ['At the age of 18, Shakespeare married 26-year-old Anne Hathaway. The consistory court of the Diocese of Worcester issued a marriage licence on 27 November 1582.']
    input_3 = ['The Battle of Halidon Hill took place on 19 July 1333 when a Scottish army under Sir Archibald Douglas attacked an English army commanded by King Edward III (reigned 1327 to 1377) and was heavily defeated. In early 1333 Edward invaded Scotland and laid siege to the strategically important border town of Berwick-upon-Tweed. A large Scottish army advanced to relieve the town.']

    inputs = [input_1, input_2, input_3]
    outputs = pipe(inputs)

    # store all features in a list
    f_container = []

    for feat in outputs:
        this_feat = np.array(feat)
        this_feat = np.reshape(this_feat, (this_feat.shape[0], -1))
        this_feat = this_feat[:, :5000]
        for i in range(10):
            f_container.append(this_feat)

    features = np.concatenate(f_container, axis=0)
    savename = os.path.join(args.savedir, f"test_text.npy")
    print(f"saving features to {savename} with shape {features.shape}")
    np.save(savename, features)
        
    print('done')


if __name__ == '__main__':
    main()