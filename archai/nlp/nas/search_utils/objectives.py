import copy

from archai.nlp.models.model_loader import load_from_args
from archai.nlp.nas.search_utils.constraints import (measure_inference_latency,
                                        measure_parameters,
                                        measure_peak_memory)
from archai.nlp.nas.converter import params_to_config


def zero_cost_objective(model_type, params, max_n_layer):
    """
    """

    def f(x):
        """
        """

        #
        model_config = load_from_args(model_type, cls_type='config').default

        #
        config = params_to_config(params, max_n_layer, x)
        model_config.update(config)

        #
        model = load_from_args(model_type, **model_config)

        print(model_config)

        n_params = measure_parameters(model)
        latency = measure_inference_latency(model)
        peak_memory = measure_peak_memory(model)

        print(latency, peak_memory)

        return n_params

    return f
