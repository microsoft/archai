"""Assess the changes in rank due to change in LR"""

import argparse
import os
import pathlib
import statistics
from ast import literal_eval

import scipy

from archai.common import delimited_text
from archai.common import utils


def main():
    default_dir = r"D:\GitHubSrc\archaiphilly\phillytools\nasbench_darts_lr0.025_wd3_b128"

    parser = argparse.ArgumentParser(description="Pytorch cifar training")
    parser.add_argument("--in-dir", default=default_dir)
    parser.add_argument("--out-dir", default=default_dir)

    args = parser.parse_args()

    parsed_metrics = delimited_text.DelimitedText()

    in_dir = pathlib.Path(utils.full_path(args.in_dir))
    assert in_dir.exists(), f"Does not exist: {in_dir}"
    metrics_filepaths = in_dir.rglob("metrics*.tsv")

    for metrics_filepath in metrics_filepaths:
        text = metrics_filepath.read_text()
        parsed_metrics.add_from_text(text, has_header=True)

    assert len(parsed_metrics) >= 1

    model_nums = [int(r) for r in parsed_metrics.get_col("model_name")]
    nasbench_acc = [statistics.mean(literal_eval(r)) for r in parsed_metrics.get_col("nasbenc101_test_acc")]
    retrain_acc = [float(r) for r in parsed_metrics.get_col("test_acc")]

    stats = list(zip(model_nums, nasbench_acc, retrain_acc))
    stats.sort(key=lambda t: t[0])

    retrain_ranks = utils.get_ranks(stats, key=lambda t: t[2])
    stats = list((i, rr, *t) for i, (t, rr) in enumerate(zip(stats, retrain_ranks)))

    corr = scipy.stats.pearsonr([t[0] for t in stats], [t[1] for t in stats])

    out_metrics = delimited_text.DelimitedText()
    out_metrics.add_from_cols_list(
        stats, header=["nasbench_rank", "rerank", "model_num", "nasbench_acc", "retrain_acc"]
    )

    rerank_filepath = os.path.join(utils.full_path(args.out_dir), "reranking.tsv")
    out_metrics.save(rerank_filepath)

    corr_filepath = os.path.join(utils.full_path(args.out_dir), "corr.txt")
    utils.write_string(corr_filepath, str(corr))


if __name__ == "__main__":
    main()
