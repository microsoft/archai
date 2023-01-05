import argparse
import os
import re
import token
import tokenize
from io import BytesIO
from tokenize import TokenInfo
from typing import List

from datasets import load_dataset, load_from_disk


def remove_single_line_comments(codestr: str) -> str:
    # split along newline characters
    lines = codestr.split("\n")
    lines_nocomm = []
    for line in lines:
        if line:
            if line[0] == "#":
                continue
            else:
                lines_nocomm.append(line)
    singlestr = "\n".join(lines_nocomm)
    return singlestr


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        default="/data/codedatasets",
        help="Full path to dataset directory",
        type=str,
    )
    parser.add_argument("--datasetname", default="only_python_dedup", help="Name of the dataset/shard")
    args = parser.parse_args()

    # load the dataset or a shard of it
    fullname = os.path.join(args.datadir, args.datasetname)
    print(f"loading {fullname}.")
    dataset = load_from_disk(fullname)
    print(f"loaded {fullname} with {dataset.num_rows} rows.")

    def cleancode(examples, indices):
        code_strings = []
        for code in examples["code"]:
            code_strings.append(remove_single_line_comments(code))
        return {"codenocomments": code_strings}

    cleandataset = dataset.map(
        cleancode,
        batched=True,
        batch_size=1000,
        num_proc=1,
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    cleandataset.save_to_disk("/data/codedatasets/only_python_dedup_nocomments")

    print("done")


if __name__ == "__main__":
    main()
