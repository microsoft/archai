import re
import tempfile
import datetime
from time import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch import profiler
from onnxruntime import InferenceSession
from efficientnet_pytorch.model import MBConvBlock


def get_utc_date():
    current_date = datetime.datetime.now()
    current_date = current_date.replace(tzinfo=datetime.timezone.utc)
    return current_date.isoformat()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_last_checkpoint(checkpoint_dir: Path):
    ckpt_dir = Path(checkpoint_dir)
    assert ckpt_dir.exists()

    checkpoints = ckpt_dir.glob('epoch=*.ckpt')
    return max(checkpoints, key=lambda x: int(re.sub('[^0-9]', '', str(x))), default=None)


def get_onnx_latency(model: torch.nn.Module, opset_version: Optional[int] = 11, img_size: Optional[int] = None):
    if not img_size:
        # Tries to infer img_size
        img_size = model.hparams['img_size']

    with tempfile.NamedTemporaryFile() as tmp:
        to_onnx(model, Path(tmp.name), img_size=img_size, opset_version=opset_version)
        latency = profile_onnx(output_path=Path(tmp.name), img_size=img_size)
    return latency['onnx_latency (ms)']


def to_onnx(model: torch.nn.Module, output_path: Path,
            img_size: Optional[int] = None,
            opset_version: Optional[int] = 11,
            overwrite: bool = True) -> None:
    model.eval().to('cpu')
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return

    if hasattr(model, 'qat') and model.qat:
        model.remove_fake_quant()

    # To ensure compatibility with older models
    if hasattr(model, 'interpolation_method') and not opset_version:
        if (model.interpolation_method != 'nearest' and model.should_interpolate):
            opset_version = 11
            print('An interpolation method different than "nearest" was detected. '
                  f'ONNX export will use opset_version = {opset_version} to ensure compatibility.')

    if not opset_version and any(isinstance(m, torch.nn.Upsample) and m.mode != 'nearest' for m in model.modules()):
        opset_version = 11
        print('An interpolation method different than "nearest" was detected. '
              f'ONNX export will use opset_version = {opset_version} to ensure compatibility.')

    # Warning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.
    # https://github.com/pytorch/pytorch/issues/73843
    # Might be an issue in quantization
    try:
        torch.onnx.export(
            model, (torch.randn(1, 3, img_size, img_size)),
            str(output_path),
            opset_version=opset_version,
            verbose=False,
            input_names=['input_0'],
            output_names=['output_0']
        )
    except Exception as e:
        print(str(e))  # So you can pipe this error message to a file for better reading
        raise e

    # print(f'ONNX model saved in {output_path}')


def profile_onnx(output_path: Path, img_size: int) -> Dict:
    onnx_session = InferenceSession(str(output_path))

    dummy_inputs = [
        torch.randn(bsz, 3, img_size, img_size) / 255
        for bsz in [1]
        for _ in range(30)
    ]

    t0 = time()

    _ = ([
        onnx_session.run(None, input_feed={'input_0': dummy_input.numpy()})[0]
        for dummy_input in dummy_inputs
    ])

    return {
        'onnx_latency (ms)': 1e3 * (time() - t0) / 30
    }


def inference_stats(model: torch.nn.Module, **inputs) -> Tuple[int, int, int]:
    # return memory usage in bytes, cpu time in us
    # We basically sum "self" time of individual ops,
    # i.e., not including child time.
    # Pytorch also has record_function which gives
    # higher CPU time, probably because it includes
    # time spent other than ops.
    # Sometime profiler also generates [memory] node
    # which has negative value of memory.
    with torch.no_grad():
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_flops=True) as prof:
            with profiler.record_function('model_inference'):
                _ = model(**inputs)
    
    t = prof.key_averages()
    self_time, self_mem, flops, ti_memory, inf_cpu, inf_mem, inf_flops = 0, 0, 0, 0, 0, 0, 0

    for ti in t:
        if ti.key == '[memory]':
            ti_memory = -ti.self_cpu_memory_usage
            continue
        if ti.key == 'model_inference':
            inf_mem = -ti.cpu_memory_usage
            inf_cpu = ti.cpu_time_total
            inf_flops = ti.flops
            continue
        self_mem += ti.self_cpu_memory_usage
        self_time += ti.self_cpu_time_total
        flops += ti.flops

    return self_mem, self_time, flops, inf_cpu


def profile_torch(model: torch.nn.Module, img_size: Optional[int] = None) -> Dict:
    model = model.to('cpu').eval()

    if not img_size:
        img_size = model.img_size if hasattr(model, 'img_size') else model.hparams['img_size']
    
    # Used for compatibility with SMP models
    predict_method = model.predict if hasattr(model.model, 'predict') else model.forward

    # Performs warmup 10 times
    with torch.no_grad():
        for _ in range(30):
            _ = predict_method(
                torch.randn(1, 3, img_size, img_size)
            )

    # Gets inference stats
    self_mem, self_time, flops, inf_cpu = inference_stats(
        predict_method, image=torch.randn(1, 3, img_size, img_size)
    )
    
    return {
        'torch_self_mem': self_mem,
        'torch_self_time': self_time,
        'flops': flops,
        'torch_inf_cpu': inf_cpu,
        'nb_parameters': count_parameters(model)
    }