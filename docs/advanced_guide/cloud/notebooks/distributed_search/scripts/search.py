import argparse
import torch
import json
from archai.discrete_search.api import SearchObjectives
from archai.discrete_search.evaluators import AvgOnnxLatency, TorchFlops
from archai.discrete_search.evaluators import TorchNumParameters
from archai.discrete_search.algos import EvolutionParetoSearch
from cnn_search_space import CNNSearchSpace
from aml_training_evaluator import AmlTrainingValAccuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to prepared dataset")
    parser.add_argument("--environment", type=str, help="name of AML environment to run the partial training jobs in")
    parser.add_argument("--compute", type=str, help="name of AML compute to run the partial training jobs on")
    parser.add_argument('--config', type=str, help='bin hexed config json info for mlclient')
    parser.add_argument("--output_dir", type=str, help="path to output data")
    args = parser.parse_args()

    environment_name = args.environment
    compute_name = args.compute
    data_dir = args.data_dir
    output_dir = args.output_dir
    config = json.loads(str(bytes.fromhex(args.config), encoding='utf-8'))

    space = CNNSearchSpace()

    search_objectives = SearchObjectives()

    search_objectives.add_constraint(
        'Number of parameters',
        TorchNumParameters(),
        constraint=(0.0, 1e6)
    )

    search_objectives.add_objective(
        # Objective function name (will be used in plots and reports)
        name='ONNX Latency (ms)',
        # ModelEvaluator object that will be used to evaluate the model
        model_evaluator=AvgOnnxLatency(input_shape=(1, 1, 28, 28), num_trials=3),
        # Optimization direction, `True` for maximization or `False` for minimization
        higher_is_better=False,
        # Whether this objective should be considered 'compute intensive' or not.
        compute_intensive=False
    )

    search_objectives.add_objective(
        name='FLOPs',
        model_evaluator=TorchFlops(torch.randn(1, 1, 28, 28)),
        higher_is_better=False,
        compute_intensive=False,
        # We may optionally add a constraint.
        # Architectures outside this range will be ignored by the search algorithm
        constraint=(0.0, 1e9)
    )

    search_objectives.add_objective(
        name='AmlTrainingValAccuracy',
        model_evaluator=AmlTrainingValAccuracy(compute_cluster_name=compute_name,
                                               environment_name=environment_name,  # AML environment name
                                               datastore_path=data_dir,  # AML datastore path
                                               models_path=output_dir,
                                               ml_config=config,
                                               training_epochs=1),
        higher_is_better=True,
        compute_intensive=True
    )

    algo = EvolutionParetoSearch(
        space,
        search_objectives,
        None,
        args.output_dir,
        num_iters=5,
        init_num_models=10,
        seed=1234,
    )

    algo.search()


if __name__ == "__main__":
    main()
