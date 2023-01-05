import argparse
import ast
import inspect
import os

from datasets import load_dataset, load_from_disk
from python_graphs import control_flow


def fn1():
    x = 0
    for i in range(5):
        x += i
    return x


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        default=os.getenv("AMLT_DATA_DIR", "/data/"),
        help="Full path to dataset directory",
        type=str,
    )
    parser.add_argument(
        "--datasetname",
        default="codeparrot-train-near-deduplication",
        help="Name of the dataset/shard",
    )
    # args = parser.parse_args()

    # # load the dataset or a shard of it
    # fullname = os.path.join(args.datadir, args.datasetname)
    # print(f"loading {fullname}.")
    # dataset = load_from_disk(fullname)
    # print(f"loaded {fullname} with {dataset.num_rows} rows.")

    # codestr = dataset["train"]["content"][0]
    # cf_graph = control_flow.get_control_flow_graph(codestr)

    # cfg_fn1 = control_flow.get_control_flow_graph(fn1)

    codestr = inspect.getsource(fn1)

    _ = ast.parse(codestr)

    print("done.")


if __name__ == "__main__":
    main()
