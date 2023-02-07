# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import sys

from overrides import overrides
from torch import nn

from archai.common import ml_utils
from archai.common.config import Config
from archai.common.ordered_dict_logger import get_global_logger
from archai.supergraph.nas.evaluater import Evaluater
from archai.supergraph.nas.model import Model
from archai.supergraph.nas.model_desc_builder import ModelDescBuilder

logger = get_global_logger()


class ManualEvaluater(Evaluater):
    @overrides
    def create_model(self, conf_eval:Config, model_desc_builder:ModelDescBuilder,
                      final_desc_filename=None, full_desc_filename=None)->nn.Module:
        # region conf vars
        dataset_name = conf_eval['loader']['dataset']['name']

        # if explicitly passed in then don't get from conf
        if not final_desc_filename:
            final_desc_filename = conf_eval['final_desc_filename']
        model_factory_spec = conf_eval['model_factory_spec']
        # endregion

        assert model_factory_spec

        return self._model_from_factory(model_factory_spec, dataset_name)

    def _model_from_factory(self, model_factory_spec:str, dataset_name:str)->Model:
        splitted = model_factory_spec.rsplit('.', 1)
        function_name = splitted[-1]

        if len(splitted) > 1:
            module_name = splitted[0]
        else:
            module_name = self._default_module_name(dataset_name, function_name)

        module = importlib.import_module(module_name) if module_name else sys.modules[__name__]
        function = getattr(module, function_name)
        model = function()

        logger.info({'model_factory':True,
                    'module_name': module_name,
                    'function_name': function_name,
                    'params': ml_utils.param_size(model)})

        return model

