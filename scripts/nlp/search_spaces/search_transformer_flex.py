# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch
from archai.nlp.objectives.parameters import NonEmbeddingParamsProxy
from archai.nlp.objectives.transformer_flex_latency import TransformerFlexOnnxLatency
from archai.nlp.objectives.transformer_flex_memory import TransformerFlexOnnxMemory
from archai.nlp.search_spaces.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Searches with TransformerFlex.")

    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        choices=["gpt2", "gpt2-flex", "mem-transformer", "opt", "transfo-xl"],
        default="gpt2",
        help="Type of model.",
    )

    parser.add_argument("-o", "--output_dir", type=str, default="", help="Output folder.")

    parser.add_argument(
        "-n",
        "--num_iters",
        type=int,
        default=10,
        help="Number of search iterations.",
    )

    parser.add_argument(
        "-inm",
        "--init_num_models",
        type=int,
        default=10,
        help="Number of initialization models.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    space = TransformerFlexSearchSpace(args.model_type)
    objectives = {
        "non_embedding_params": NonEmbeddingParamsProxy(),
        "onnx_latency": TransformerFlexOnnxLatency(space),
        "onnx_memory": TransformerFlexOnnxMemory(space),
    }

    algo = EvolutionParetoSearch(
        space,
        objectives,
        None,
        args.output_dir,
        num_iters=args.num_iters,
        init_num_models=args.init_num_models,
        seed=args.seed,
    )
    algo.search()
