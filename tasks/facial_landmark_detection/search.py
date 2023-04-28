# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from overrides.overrides import overrides

from archai.common.common import logger
from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.evaluators.ray import RayParallelEvaluator

import train as model_trainer
from dataset import FaceLandmarkDataset
from latency import AvgOnnxLatency
from search_space import ConfigSearchSpaceExt


class ValidationErrorEvaluator(ModelEvaluator):
    def __init__(self, args) -> None:
        self.args = args

    @overrides
    def evaluate(self, model, dataset_provider, budget=None) -> float:
        logger.info(f"evaluating {model.archid}")

        val_error = model_trainer.train(self.args, model.arch)
        if math.isnan(val_error):
            logger.info(
                f"Warning: model {model.archid} has val_error NaN. Set to 10000.0 to avoid corrupting the Pareto front."
            )
            val_error = 10000.0
        return val_error


class OnnxLatencyEvaluator(ModelEvaluator):
    def __init__(self, args) -> None:
        self.args = args
        self.latency_evaluator = AvgOnnxLatency(
            input_shape=(1, 3, 128, 128),
            num_trials=self.args.num_latency_measurements,
            num_input=self.args.num_input_per_latency_measurement,
        )

    @overrides
    def evaluate(self, model, dataset_provider, budget=None) -> float:
        return self.latency_evaluator.evaluate(model)


class SearchFaceLandmarkModels:
    def __init__(self) -> None:
        super().__init__()

        config_parser = ArgumentParser(conflict_handler="resolve", description="NAS for Facial Landmark Detection.")
        config_parser.add_argument(
            "--config", required=True, type=Path, help="YAML config file specifying default arguments"
        )

        parser = ArgumentParser(conflict_handler="resolve", description="NAS for Facial Landmark Detection.")
        parser.add_argument("--data_path", required=False, type=Path)
        parser.add_argument("--output_dir", required=True, type=Path)
        parser.add_argument("--num_jobs_per_gpu", required=False, type=int, default=1)

        def _parse_args_from_config(parser_to_use):
            args_config, remaining = config_parser.parse_known_args()
            if args_config.config:
                with open(args_config.config, "r") as f:
                    cfg = yaml.safe_load(f)
                    # The usual defaults are overridden if a config file is specified.
                    parser_to_use.set_defaults(**cfg)
            # The parser to be used parses the rest of the known command line args.
            args, _ = parser_to_use.parse_known_args(remaining)

            return args

        # parse twice to get the search args and trainer args
        self.search_args = _parse_args_from_config(parser)
        self.trainer_args = _parse_args_from_config(model_trainer.get_args_parser())

    def search(self):
        dataset = FaceLandmarkDataset(self.trainer_args.data_path)
        ss = ConfigSearchSpaceExt(self.search_args, num_classes=dataset.num_landmarks)

        search_objectives = SearchObjectives()
        search_objectives.add_objective(
            "Onnx_Latency_(ms)", OnnxLatencyEvaluator(self.search_args), higher_is_better=False, compute_intensive=False
        )
        search_objectives.add_objective(
            "Partial_Training_Validation_Error",
            RayParallelEvaluator(
                ValidationErrorEvaluator(self.trainer_args),
                num_gpus=1.0 / self.search_args.num_jobs_per_gpu,
                max_calls=1,
            ),
            higher_is_better=False,
            compute_intensive=True,
        )

        algo = EvolutionParetoSearch(
            search_space=ss,
            search_objectives=search_objectives,
            output_dir=self.search_args.output_dir,
            num_iters=self.search_args.num_iters,
            init_num_models=self.search_args.init_num_models,
            num_random_mix=self.search_args.num_random_mix,
            max_unseen_population=self.search_args.max_unseen_population,
            mutations_per_parent=self.search_args.mutations_per_parent,
            num_crossovers=self.search_args.num_crossovers,
            seed=self.search_args.seed,
            save_pareto_model_weights=False,
        )

        search_results = algo.search()

        results_df = search_results.get_search_state_df()
        ids = results_df.archid.values.tolist()
        if len(set(ids)) > len(ids):
            print("Duplicated models detected in nas results. This is not supposed to happen.")
            assert False

        configs = []
        for archid in ids:
            cfg = ss.config_all[archid]
            configs.append(cfg)
        config_df = pd.DataFrame({"archid": ids, "config": configs})
        config_df = results_df.merge(config_df)

        output_csv_name = "-".join(["search-results", datetime.now().strftime("%Y%m%d-%H%M%S"), ".csv"])
        output_csv_path = os.path.join(self.search_args.output_dir, output_csv_name)
        config_df.to_csv(output_csv_path)
        return


def _main() -> None:
    search = SearchFaceLandmarkModels()
    search.search()


if __name__ == "__main__":
    _main()
