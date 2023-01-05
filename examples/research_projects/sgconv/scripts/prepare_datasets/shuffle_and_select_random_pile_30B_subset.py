from datasets import DatasetDict, concatenate_datasets, load_from_disk


def main():
    print("loading big dataset.")
    dataset = load_from_disk("/data/dataroot/the_pile_encoded_2048")
    print("finished loading.")

    shuffled_dataset = dataset.shuffle(seed=42)
    print("finished shuffling.")

    # with 2048 sequence length you get
    # 14648438 sequences to total 30B tokens
    small_dataset = shuffled_dataset["train"].select(range(14648438))
    print("finished selecting random subset")

    print("saving to disk small dataset")
    small_dataset = DatasetDict({"train": small_dataset})
    small_dataset.save_to_disk("/data1/the_pile_30B_encoded_2048")
    print("finished saving.")


if __name__ == "__main__":
    main()
