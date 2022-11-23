# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch
from archai.nlp.objectives.decoder_param import NonEmbeddingParamsProxy
from archai.nlp.objectives.transformer_flex_latency import TransformerFlexOnnxLatency
from archai.nlp.objectives.transformer_flex_memory import TransformerFlexOnnxMemory
from archai.nlp.search_spaces.transformer_flex.search_space import (
    TransformerFlexSearchSpace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Searches with TransformerFlex.")

    parser.add_argument(
        "-mt", "--model_type", type=str, choices=["gpt2", "gpt2-flex"], default="gpt2", help="Type of model."
    )

    parser.add_argument("-od", "--output_dir", type=str, default="", help="Output folder.")

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

    algo = EvolutionParetoSearch(space, objectives, None, args.output_dir)
    algo.search()
