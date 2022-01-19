import copy

from archai.nlp.nas.converter import params_to_config
from archai.nlp.nas.constraints import measure_parameters

from archai.nlp.common.lazy_loader import load_from_args


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

        n_params = measure_parameters(model)

        return n_params['total']

    return f
