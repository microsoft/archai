import numpy as np
from datasets import load_from_disk


def main():
    dataset = load_from_disk("/data/codedatasets/only_python_dedup")
    num_to_sample = int(dataset.num_rows // 10)

    rng = np.random.default_rng(seed=42)
    rints = rng.choice(dataset.num_rows, size=num_to_sample)
    random_subset = dataset.select(rints)

    random_subset.save_to_disk("/data/codedatasets/only_python_dedup_random_10")


if __name__ == "__main__":
    main()
