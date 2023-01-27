# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.api.search_objectives import SearchObjectives
from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch
from archai.discrete_search.evaluators.nlp.parameters import NonEmbeddingParamsProxy
from archai.discrete_search.evaluators.nlp.transformer_flex_latency import (
    TransformerFlexOnnxLatency,
)
from archai.discrete_search.evaluators.nlp.transformer_flex_memory import (
    TransformerFlexOnnxMemory,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Searches with Transformer-Flex.")

    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        choices=["codegen", "gpt2", "gpt2-flex", "mem-transformer", "opt", "transfo-xl"],
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
        "-nrm",
        "--num_random_mix",
        type=int,
        default=5,
        help="Number of random models to mix with the population in each iteration.",
    )

    parser.add_argument(
        "-mup",
        "--max_unseen_population",
        type=int,
        default=100,
        help="Maximum number of unseen models in each iteration.",
    )

    parser.add_argument(
        "-mpp",
        "--mutations_per_parent",
        type=int,
        default=1,
        help="Number of distinct mutations generated for each Pareto frontier member.",
    )

    parser.add_argument(
        "-nc",
        "--num_crossovers",
        type=int,
        default=5,
        help="Total number of crossovers generated per iteration.",
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

    search_objectives = SearchObjectives()
    search_objectives.add_objective(
        "non_embedding_params",
        NonEmbeddingParamsProxy(),
        higher_is_better=True,
        compute_intensive=False,
        constraint=(1e6, 1e9),
    )
    search_objectives.add_objective(
        "onnx_latency",
        TransformerFlexOnnxLatency(space),
        higher_is_better=False,
        compute_intensive=False,
    )
    search_objectives.add_objective(
        "onnx_memory",
        TransformerFlexOnnxMemory(space),
        higher_is_better=False,
        compute_intensive=False,
    )

    algo = EvolutionParetoSearch(
        space,
        search_objectives,
        None,
        args.output_dir,
        num_iters=args.num_iters,
        init_num_models=args.init_num_models,
        num_random_mix=args.num_random_mix,
        max_unseen_population=args.max_unseen_population,
        mutations_per_parent=args.mutations_per_parent,
        num_crossovers=args.num_crossovers,
        seed=args.seed,
    )
    algo.search()
