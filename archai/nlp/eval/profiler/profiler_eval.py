import torch
import torch.nn as nn
from archai.nlp.eval.profiler.profiler_model import ProfilerModel
from archai.nlp.eval.profiler.profiler_utils import macs_to_string, number_to_string, params_to_string



def get_model_profile(
    model,
    input_shape=None,
    args=[],
    kwargs={},
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=1,
    as_string=True,
    output_file=None,
    ignore_modules=None,
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Example:

    .. code-block:: python

        model = torchvision.models.alexnet()
        batch_size = 256
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 3, 224, 224)))

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the latency of each module. Defaults to 1.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    prof = ProfilerModel(model)
    model.eval()

    if input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        try:
            input = torch.ones(()).new_empty(
                (*input_shape,
                 ),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            input = torch.ones(()).new_empty((*input_shape, ))

        args = [input]
    assert (len(args) > 0) or (len(kwargs) > 0), "args and/or kwargs must be specified if input_shape is None"

    for _ in range(warm_up):
        if kwargs:
            _ = model(*args, **kwargs)
        else:
            _ = model(*args)
    prof.start_profile(ignore_list=ignore_modules)

    if kwargs:
        _ = model(*args, **kwargs)
    else:
        _ = model(*args)

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=warm_up,
                                 module_depth=module_depth,
                                 top_modules=top_modules,
                                 detailed=detailed,
                                 output_file=output_file)

    prof.end_profile()
    if as_string:
        return number_to_string(flops), macs_to_string(macs), params_to_string(params)

    return flops, macs, params
