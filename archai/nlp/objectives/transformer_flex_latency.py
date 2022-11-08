import copy
import os
import types
from pathlib import Path
import tempfile
from typing import Optional, Sized

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.transformers import quantize_helper
from archai.nlp.onnx.export_utils import prepare_model_for_onnx
from overrides import overrides
import torch

from archai.discrete_search import ArchaiModel, DatasetProvider, Objective
from archai.nlp.onnx.optimization import optimize_onnx
from archai.nlp.onnx.onnx_forward import gpt2_onnx_forward
from archai.nlp.onnx.export import export_to_onnx

# ONNX-loading constants
OMP_NUM_THREADS = 1
OMP_WAIT_POLICY = 'ACTIVE'

# Constants available in onnxruntime
# that enables performance optimization
os.environ['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
os.environ['OMP_WAIT_POLICY'] = OMP_WAIT_POLICY


class TransformerFlexOnnxLatency(Objective):
    higher_is_better: bool = False

    def __init__(self, search_space) -> None:
        assert search_space.arch_type in ['gpt2', 'gpt2-flex']
        self.search_space = search_space
    
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
            export_to_onnx(model, Path(tmp.name), task='causal-lm',
                           use_past=True, share_weights=True, opset=11)
            optimize_onnx(Path(tmp.name), model.config)
            

