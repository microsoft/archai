from archai.discrete_search.search_spaces.nlp.transformer_flex.search_space import TransformerFlexSearchSpace
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.evaluators.nlp.parameters import NonEmbeddingParamsProxy
from archai.discrete_search.evaluators.nlp.transformer_flex_latency import TransformerFlexOnnxLatency
from archai.discrete_search.evaluators.nlp.transformer_flex_memory import TransformerFlexOnnxMemory
from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="nas_output", help="path to output data")
    args = parser.parse_args()

    space = TransformerFlexSearchSpace("gpt2")

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
        num_iters=5,
        init_num_models=10,
        save_pareto_model_weights=False,
        seed=1234,
    )

    algo.search()

if __name__ == "__main__":
    main()