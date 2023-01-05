from collections import defaultdict

from datasets import load_from_disk


def main():
    dataset = load_from_disk("/data/codedatasets/only_python")

    dups_inds = defaultdict(list)

    def count_orig_dups(examples, indices):
        for index, orig_index in enumerate(indices):
            dups_inds[examples["code"][index]].append(orig_index)

    _ = dataset.map(count_orig_dups, with_indices=True, batched=True, batch_size=1000, num_proc=1)

    dups_freqs = []
    for k, v in dups_inds.items():
        if len(v) > 1:
            dups_freqs.append(len(v))

    ind_groups_by_dup = [inds for inds in dups_inds.values()]
    dedup_indices = [inds[0] for inds in ind_groups_by_dup]
    dedup_dataset = dataset.select(dedup_indices)
    dedup_dataset.save_to_disk("/data/codedatasets/only_python_dedup")

    print("done")


if __name__ == "__main__":
    main()
