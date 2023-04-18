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
import torch
import train as model_trainer
import yaml
from discrete_search_space_mnv2_config import (
    ConfigSearchSpaceExt,
    _create_model_from_csv,
    _load_pretrain_weight,
)
from latency_measurement import AvgOnnxLatency
from overrides.overrides import overrides

from archai.common.common import logger
from archai.discrete_search.algos.evolution_pareto import EvolutionParetoSearch
from archai.discrete_search.api.model_evaluator import ModelEvaluator
from archai.discrete_search.api.search_objectives import SearchObjectives


#hardcoded need to get as a parameter
NUM_LANDMARK_CLASSES = 140

def convert_args_dict_to_list(d):
    if d is None:
        return []

    new_list = []
    for key, val in d.items():
        new_list.append(f"--{key}")
        new_list.append(f"{val}")

    return new_list

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
        self.latency_evaluator = AvgOnnxLatency(input_shape=(1, 3, 128, 128), num_trials=self.args.nas_num_latency_measurements, num_input=self.args.nas_num_input_per_latency_measurement)

    @overrides
    def evaluate(self, model, dataset_provider, budget = None) -> float:
        return self.latency_evaluator.evaluate(model)
        
class NASLandmarks():
    def __init__(self) -> None:
        super().__init__()

        config_parser = ArgumentParser(conflict_handler="resolve", description='NAS on Face Tracking.')
        config_parser.add_argument("--nas_config", required=True, type=Path, help='YAML config file specifying default arguments')
        
        parser = ArgumentParser(conflict_handler="resolve", description='NAS on Face Tracking.')
        parser.add_argument("--nas_output_dir", required=True, type=Path)
        parser.add_argument("--nas_search_backbone", type=str, help='backbone to used use for config search')
        parser.add_argument('--nas_num_jobs_per_gpu', required=False, type=int, default=1)

        finalize_group = parser.add_argument_group('Finalize parameters')
        finalize_group.add_argument('--nas_finalize_archid', required=False, type=str, help='archid for the model to be finalized')
        finalize_group.add_argument("--nas_finalize_models_csv", required='--nas_finalize_archid' in sys.argv, type=str, help='csv file output from the search stage')
        finalize_group.add_argument("--nas_finalize_pretrained_weight_file", required=False, type=str, help='weight file from pretraining')

        qat_group = parser.add_argument_group('QAT parameters')
        qat_group.add_argument("--nas_use_tvmodel", action='store_true', help='Use Torchvision model')
        qat_group.add_argument("--nas_qat", action='store_true', help='Use model ready for quantization aware training')
        qat_group.add_argument("--nas_load_nonqat_weights", action='store_true', help='Use weights from previous training without QAT')

        def _parse_args_from_config():
            args_config, remaining = config_parser.parse_known_args()
            if args_config.nas_config:
                with open(args_config.nas_config, 'r') as f:
                    cfg = yaml.safe_load(f)
                    # The usual defaults are overridden if a config file is specified.
                    parser.set_defaults(**cfg)

            # The main arg parser parses the rest of the known command line args.
            args, remaining_args = parser.parse_known_args(remaining)
            # Args in the config file will be returned as a list of strings to
            # be further used by LitLandmarksTrainer
            remaining_args = remaining_args + convert_args_dict_to_list(cfg) if cfg else None

            return args, remaining_args

        self.nas_args, remaining_args = _parse_args_from_config()
        self.lit_args, _ = model_trainer.get_args_parser().parse_known_args(remaining_args)

#TBD
#        print (f"Loading dataset from {self.trainer_args.data_dir}")
        self.datamodule = None

    def search(self):
        if (self.nas_args.nas_search_backbone == 'mobilenetv2'):
            ss = ConfigSearchSpaceExt (self.nas_args, num_classes = NUM_LANDMARK_CLASSES)
        else:
            print(f"self.nas_args.nas_search_backbone: {self.nas_args.nas_search_backbone}, not supported")
            assert (False)

        search_objectives = SearchObjectives()
        search_objectives.add_objective(
                'Partial training Validation Accuracy',
                AccuracyEvaluator(self.lit_args),
                higher_is_better=False,
                compute_intensive=True)
        search_objectives.add_objective(
                "onnx_latency (ms)",
                OnnxLatencyEvaluator(self.nas_args),
                higher_is_better=False,
                compute_intensive=False)

        algo = EvolutionParetoSearch(
            search_space=ss,        
            search_objectives=search_objectives,
            output_dir=self.nas_args.nas_output_dir,
            num_iters=self.nas_args.nas_num_iters,
            init_num_models=self.nas_args.nas_init_num_models,
            num_random_mix=self.nas_args.nas_num_random_mix,
            max_unseen_population=self.nas_args.nas_max_unseen_population,
            mutations_per_parent=self.nas_args.nas_mutations_per_parent,
            num_crossovers=self.nas_args.nas_num_crossovers,
            seed=self.nas_args.seed,
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
        output_csv_path = os.path.join(self.nas_args.nas_output_dir, output_csv_name)
        config_df.to_csv(output_csv_path)
        return


def _main() -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # To overcome 'imgcodecs: OpenEXR codec is disabled' error

    nas = NASLandmarks()
    if (None == nas.nas_args.nas_finalize_archid):
        #args = Namespace(**vars(nas.nas_args), **vars(nas.lit_args))
        nas.search()
    else:
        model = _create_model_from_csv (
            nas.nas_args.nas_finalize_archid,
            nas.nas_args.nas_finalize_models_csv,
            num_classes=NUM_LANDMARK_CLASSES)

        print(f'Loading weights from {str(nas.nas_args.nas_finalize_pretrained_weight_file)}')
        if (not nas.nas_args.nas_load_nonqat_weights):
            if (nas.nas_args.nas_finalize_pretrained_weight_file is not None) :
                model = _load_pretrain_weight(nas.nas_args.nas_finalize_pretrained_weight_file, model)

        if (nas.nas_args.nas_use_tvmodel):
            model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(model.last_channel, NUM_LANDMARK_CLASSES))

        # Load pretrained weights after fixing classifier as the weights match the exact network architecture
        if (nas.nas_args.nas_load_nonqat_weights):
            assert os.path.exists(nas.nas_args.nas_finalize_pretrained_weight_file)
            print(f'Loading weights from previous non-QAT training {nas.nas_args.nas_finalize_pretrained_weight_file}')
            model.load_state_dict(torch.load(nas.nas_args.nas_finalize_pretrained_weight_file))

        val_error = model_trainer.train(nas.lit_args, model)
        print(f"Final validation error for model {nas.nas_args.nas_finalize_archid}: {val_error}")

if __name__ == "__main__":
    _main()