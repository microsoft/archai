import pathlib

from datasets import DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm


def main():
    shard_path = pathlib.Path("/data/dataroot/the_pile_sharded_encoded")
    dirs = list(shard_path.iterdir())

    datasets = []
    for dir in tqdm(dirs):
        datasets.append(load_from_disk(dir)["train"])

    concat_dataset = concatenate_datasets(datasets)
    concat_dataset = DatasetDict({"train": concat_dataset})
    concat_dataset.save_to_disk("/data/dataroot/the_pile_encoded_2048")

    print("done")


if __name__ == "__main__":
    main()
