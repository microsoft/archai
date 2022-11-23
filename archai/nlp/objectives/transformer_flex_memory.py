import copy
import os
from pathlib import Path
import tempfile
import timeit
from typing import Optional, Dict, Tuple, Any, Sized
import types

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.transformers import quantize_helper
from archai.nlp.onnx.export_utils import prepare_model_for_onnx
from overrides import overrides
import numpy as np
import torch

from archai.discrete_search import ArchaiModel, DatasetProvider, Objective
from archai.nlp.search_spaces.transformer_flex.search_space import TransformerFlexSearchSpace
from archai.nlp.onnx.optimization import optimize_onnx
from archai.nlp.onnx.config_utils.onnx_config_base import OnnxConfig
from archai.nlp.onnx.onnx_forward import gpt2_onnx_forward
from archai.nlp.onnx.export import export_to_onnx

# ONNX-loading constants
OMP_NUM_THREADS = 1
OMP_WAIT_POLICY = 'ACTIVE'

# Constants available in onnxruntime
# that enables performance optimization
os.environ['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
os.environ['OMP_WAIT_POLICY'] = OMP_WAIT_POLICY


class TransformerFlexOnnxMemory(Objective):
    higher_is_better: bool = False

    def __init__(self, search_space: TransformerFlexSearchSpace,
                 batch_size: int = 1,
                 seq_len: int = 192,
                 past_seq_len: int = 0,
                 n_trials: int = 1, use_median: bool = False) -> None:
        assert search_space.arch_type in ['gpt2', 'gpt2-flex']
        self.search_space = search_space
        
        # Benchmark settings
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.past_seq_len = past_seq_len
        self.n_trials = n_trials
        self.use_median = use_median
        
        # Onnx runtime session options
        self.options = SessionOptions()
        self.options.intra_op_num_threads = OMP_NUM_THREADS
        self.options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    
    def _load_and_prepare(self, config) -> torch.nn.Module:
        config = copy.deepcopy(config)
        config['use_cache'] = True

        model = self.search_space._load_model_from_config(config)
        return prepare_model_for_onnx(model, self.search_space.arch_type)

    @overrides
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider,
                 budget: Optional[float] = None) -> float:
        model = self._load_and_prepare(arch.metadata['config'])
        
        with tempfile.NamedTemporaryFile() as tmp:

            onnx_config = export_to_onnx(
                model, Path(tmp.name), task='causal-lm',
                use_past=True, share_weights=True, opset=11
            )
            optimize_onnx(tmp.name, onnx_config, opt_level=0)

            disk_memory = os.path.getsize(tmp.name)
            disk_memory_mb = disk_memory / (1024 ** 2)

            return disk_memory_mb

