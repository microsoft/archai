# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import pathlib

from archai.supergraph.nas.model_desc import ModelDesc
from archai.supergraph.nas.vis_model_desc import draw_model_desc
from archai.common import utils


def main():
    parser = argparse.ArgumentParser(description="Visualize model description")
    parser.add_argument(
        "-f", "--model-desc-file", type=str, default="models/final_model_desc5.yaml", help="Model desc file"
    )
    args, extra_args = parser.parse_known_args()

    model_desc_filepath = utils.full_path(args.model_desc_file)
    model_desc = ModelDesc.load(model_desc_filepath)

    out_file = pathlib.Path(model_desc_filepath).with_suffix("")

    draw_model_desc(model_desc, str(out_file))


if __name__ == "__main__":
    main()
