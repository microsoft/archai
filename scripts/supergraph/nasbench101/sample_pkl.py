import pickle

from archai.supergraph.utils import utils


def main():
    in_dataset_file = utils.full_path("~/dataroot/nasbench_ds/nasbench_full.tfrecord.pkl")
    out_dataset_file = utils.full_path("~/dataroot/nasbench_ds/nasbench101_sample.tfrecord.pkl")

    with open(in_dataset_file, "rb") as f:
        records = pickle.load(f)

    sampled_indices = set()
    adj_samples = 1000
    for i in [0, 4000, 40000, len(records) - 1 - adj_samples + 1]:
        sampled_indices = sampled_indices.union([i + k for k in range(adj_samples)])

    sampled_hashes = set(records[i][0] for i in sorted(list(sampled_indices)))
    sampled = [r for r in records if r[0] in sampled_hashes]
    with open(out_dataset_file, "wb") as f:
        pickle.dump(sampled, f)


if __name__ == "__main__":
    main()
