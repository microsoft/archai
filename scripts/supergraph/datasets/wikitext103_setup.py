import argparse
import os
from collections import Counter

import numpy as np

from archai.supergraph.utils import utils


class _Corpus:
    def __init__(self, word2idx, path):
        self.word2idx = word2idx
        self.train = self.encode_file(path % "train")
        self.valid = self.encode_file(path % "valid")
        self.test = self.encode_file(path % "test")

    def encode_file(self, path):
        tokens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for w in words:
                    tokens.append(self.word2idx[w])
        return np.array(tokens)


def main():
    parser = argparse.ArgumentParser(description="Pytorch cifar training")
    parser.add_argument("--datadir", default="~/dataroot/textpred/wikitext-103")  # .token files from original dataset
    parser.add_argument(
        "--cachedir", default="~/dataroot/textpred/wikitext-103-sorted-vocab"
    )  # this should have 'wikitext-103-sorted_vocab.npy' file

    args = parser.parse_args()

    args.datadir = utils.full_path(args.datadir)
    args.cachedir = utils.full_path(args.cachedir)

    # For compatibility, we use a precomputed vocab sorted from most to least occurence in the training set
    sorted_vocab_filepath = os.path.join(args.cachedir, "sorted_vocab.npy")
    idx2word = np.load(sorted_vocab_filepath, allow_pickle=True)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    assert len(word2idx) == 267735
    # Might take 5-10 minutes to run
    corpus = _Corpus(word2idx, os.path.join(args.datadir, "wiki.%s.tokens"))

    train_counts = Counter(corpus.train)
    for i, (token, count) in enumerate(train_counts.most_common()):  # Check that our vocab is indeed sorted
        assert count == train_counts[i]

    for k in "train", "valid", "test":
        sorted_file_path = os.path.join(args.cachedir, "sorted_" + k + ".npy")
        np.save(sorted_file_path, getattr(corpus, k).astype(np.int32))


if __name__ == "__main__":
    main()
