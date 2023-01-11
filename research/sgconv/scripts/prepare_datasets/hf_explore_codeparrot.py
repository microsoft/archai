from collections import defaultdict

from datasets import load_from_disk


def main():
    dataset = load_from_disk("/data/codedatasets/codeparrot-train-near-deduplication")
    dataset = dataset["train"]
    print("done")

    dups_inds = defaultdict(list)

    def count_orig_dups(examples, indices):
        for codestr, index in zip(examples["content"], indices):
            dups_inds[codestr].append(index)

    _ = dataset.map(count_orig_dups, with_indices=True, batched=True, batch_size=1000, num_proc=1)

    dups_freqs = []
    for k, v in dups_inds.items():
        if len(v) > 1:
            dups_freqs.append(len(v))

    print("done")


if __name__ == "__main__":
    main()
