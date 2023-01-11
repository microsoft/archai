from collections import defaultdict

import numpy as np
from datasets import load_dataset, load_from_disk


def main():
    dataset = load_dataset("codeparrot/github-jupyter-text-code-pairs")

    # dataset = load_from_disk("/data/codedatasets/github-jupyter-text-code-pairs")
    dataset = dataset["train"]
    print("done")

    dups_inds = defaultdict(list)

    def count_orig_dups(examples, indices):
        for index, orig_index in enumerate(indices):
            concat_str = examples["markdown"][index] + " \n " + examples["code"][index]
            dups_inds[concat_str].append(orig_index)

    _ = dataset.map(count_orig_dups, with_indices=True, batched=True, batch_size=1000, num_proc=1)

    dups_freqs = []
    for k, v in dups_inds.items():
        if len(v) > 1:
            dups_freqs.append(len(v))

    print("done")


if __name__ == "__main__":
    main()
