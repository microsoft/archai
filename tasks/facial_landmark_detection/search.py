import sys

if ('--debug' in sys.argv):
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger")
    debugpy.wait_for_client()

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

import train as model_trainer
from dataset import FaceLandmarkDataset
from latency import AvgOnnxLatency
from search_space import ConfigSearchSpaceExt
class AccuracyEvaluator(ModelEvaluator):
    
    def __init__(self, lit_args) -> None:
        self.lit_args = lit_args

    @overrides
    def evaluate(self, model, dataset_provider, budget = None) -> float:
        logger.info(f"evaluating {model.arch.archid}")

        val_error = model_trainer.train(self.lit_args, model.arch) #, model.arch.archid, dataset_provider)
        if (math.isnan(val_error)):
            logger.info(f"Warning: model {model.arch.archid} has val_error NaN. Set to 10000.0 to avoid corrupting the Pareto front.")
            val_error = 10000.0
        return val_error

class OnnxLatencyEvaluator(ModelEvaluator):

    def __init__(self, args) -> None:
        self.args = args
        self.latency_evaluator = AvgOnnxLatency(input_shape=(1, 3, 128, 128), num_trials=self.args.num_latency_measurements, num_input=self.args.num_input_per_latency_measurement)

    @overrides
    def evaluate(self, model, dataset_provider, budget = None) -> float:
        return self.latency_evaluator.evaluate(model)
        
class SearchFaceLandmarkModels():
    def __init__(self) -> None:
        super().__init__()

        config_parser = ArgumentParser(conflict_handler="resolve", description='NAS for Facial Landmark Detection.')
        config_parser.add_argument("--config", required=True, type=Path, help='YAML config file specifying default arguments')
        
        parser = ArgumentParser(conflict_handler="resolve", description='NAS for Facial Landmark Detection.')
        parser.add_argument("--output_dir", required=True, type=Path)
        parser.add_argument('--num_jobs_per_gpu', required=False, type=int, default=1)

        def convert_args_dict_to_list(d):
            if d is None:
                return []

            new_list = []
            for key, val in d.items():
                new_list.append(f"--{key}")
                new_list.append(f"{val}")

            return new_list

        def _parse_args_from_config():
            args_config, remaining = config_parser.parse_known_args()
            if args_config.config:
                with open(args_config.config, 'r') as f:
                    cfg = yaml.safe_load(f)
                    # The usual defaults are overridden if a config file is specified.
                    parser.set_defaults(**cfg)

            # The main arg parser parses the rest of the known command line args.
            args, remaining_args = parser.parse_known_args(remaining)
            # Args in the config file will be returned as a list of strings to
            # be further used by the trainer
            remaining_args = remaining_args + convert_args_dict_to_list(cfg) if cfg else None

            return args, remaining_args

        self.search_args, remaining_args = _parse_args_from_config()
        self.trainer_args, _ = model_trainer.get_args_parser().parse_known_args(remaining_args)

    def search(self):

        dataset = FaceLandmarkDataset (self.trainer_args.data_path)
        ss = ConfigSearchSpaceExt (self.search_args, num_classes = dataset.num_landmarks)

        search_objectives = SearchObjectives()
        search_objectives.add_objective(
                'Partial training Validation Accuracy',
                AccuracyEvaluator(self.trainer_args),
                higher_is_better=False,
                compute_intensive=True)
        search_objectives.add_objective(
                "onnx_latency (ms)",
                OnnxLatencyEvaluator(self.search_args),
                higher_is_better=False,
                compute_intensive=False)

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
            save_pareto_model_weights = False)

        search_results = algo.search()

        results_df = search_results.get_search_state_df()
        ids = results_df.archid.values.tolist()
        if (len(set(ids)) > len(ids)):
            print("Duplidated models detected in nas results. This is not supposed to happen.")
            assert (False)

        configs = []
        for archid in ids:
            cfg = ss.config_all[archid]
            configs.append(cfg)
        config_df = pd.DataFrame({'archid' : ids, 'config' : configs})
        config_df = results_df.merge(config_df)

        output_csv_name = '-'.join(['search',
                                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                                    'output.csv'])
        output_csv_path = os.path.join(self.search_args.output_dir, output_csv_name)
        config_df.to_csv(output_csv_path)
        return


def _main() -> None:

    search = SearchFaceLandmarkModels()
    search.search()

    ###To be moved to trainder
    """
        model = _create_model_from_csv (
            nas.search_args.nas_finalize_archid,
            nas.search_args.nas_finalize_models_csv,
            num_classes=NUM_LANDMARK_CLASSES)

        print(f'Loading weights from {str(nas.search_args.nas_finalize_pretrained_weight_file)}')
        if (not nas.search_args.nas_load_nonqat_weights):
            if (nas.search_args.nas_finalize_pretrained_weight_file is not None) :
                model = _load_pretrain_weight(nas.search_args.nas_finalize_pretrained_weight_file, model)

        if (nas.search_args.nas_use_tvmodel):
            model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(model.last_channel, NUM_LANDMARK_CLASSES))

        # Load pretrained weights after fixing classifier as the weights match the exact network architecture
        if (nas.search_args.nas_load_nonqat_weights):
            assert os.path.exists(nas.search_args.nas_finalize_pretrained_weight_file)
            print(f'Loading weights from previous non-QAT training {nas.search_args.nas_finalize_pretrained_weight_file}')
            model.load_state_dict(torch.load(nas.search_args.nas_finalize_pretrained_weight_file))

        val_error = model_trainer.train(nas.trainer_args, model)
        print(f"Final validation error for model {nas.search_args.nas_finalize_archid}: {val_error}")
    """

if __name__ == "__main__":
    _main()