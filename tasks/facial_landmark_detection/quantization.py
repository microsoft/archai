#%%
from ast import walk
from typing import Any
import torch
import os
import copy
import numpy as np
from face_synthetics_training.training.datasets.synthetics import CogFaceSynthetics
from face_synthetics_training.training.landmarks2 import data_module
from face_synthetics_training.training.landmarks2.lit_landmarks import LitLandmarksTrainer, unnormalize_coordinates, landmarks_error
from face_synthetics_training.training.landmarks2.data_module import SyntheticsDataModule
from face_synthetics_training.training.landmarks2.nas.mobilenetv2 import InvertedResidual, MobileNetV2
from torchvision.models.quantization.mobilenetv2 import mobilenet_v2
import torchvision.models as models
import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean
from queue import Queue
import torchvision.models.quantization as tvqntmodels

import tempfile
from pathlib import Path
from face_synthetics_training.training.landmarks2.nas.utils import to_onnx, get_model_flops, profile_onnx, get_model_latency_1cpu, get_time_elapsed
from onnxruntime import InferenceSession
from time import time
import statistics

# This script handles static quantization of PyTorch MobilenetV2 model.
# However, it was found that ONNX export of quantized models is not supported.
# So, the reduction in latency achieved by static quantizing PyTorch model was subpar
# compared to just converting the base model to ONNX. Still, static quantization of 
# converted ONNX model using ONNX APIs yielded better latency. So, going with that approach.
# Leaving this code intact for any future reference.

def save_model(model:torch.nn.Module, model_filepath:Path):
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model

# def get_time_elapsed (model, img_size: int, onnx: bool = False) -> float :
#   #sanity check to make sure we are consisent
#     num_input = 10
#     input_img = torch.randn(1, 3, img_size, img_size)

#     if (onnx) :
#         with tempfile.NamedTemporaryFile() as tmp:
#             output_path = Path(tmp.name)
#             to_onnx(model, output_path, img_size=(img_size, img_size))
#             onnx_session = InferenceSession(str(output_path))

#     def meausre_func() : 
#         t0 = time()
#         for _ in range(num_input):
#             if (onnx):
#                 input_name = onnx_session.get_inputs()[0].name
#                 onnx_session.run(None, input_feed={input_name: input_img.numpy()})[0]
#             else:
#                 pred = model.forward(input_img)
#         time_measured = 1e3 * (time() - t0) / num_input
#         return time_measured

#     while True:
#         time_measured_all = [meausre_func() for _ in range (10)]
#         time_measured_avg = statistics.mean(time_measured_all)
#         time_measured_std = statistics.stdev(time_measured_all)
#         if (time_measured_std < time_measured_avg * 0.1):
#             break

#     return time_measured_avg, time_measured_std

def get_1cpu_latency(model_path:str, onnx=False):
    # model = torch.load(model_path)
    # model.to('cpu').eval()
    with torch.no_grad():
        print(get_model_latency_1cpu(
            model_path,
            img_size=192,
            cpu=1,
            onnx=onnx,
            num_input=128
        ))

def calculate_latency(model:torch.nn.Module, img_size:int):
    img_size = [img_size, img_size]

    dummy_inputs = [
        torch.randn(bsz, 3, *img_size[::-1]) % 255
        for bsz in [1]
        for _ in range(30)
    ]

    t0 = time()
    _ = ([
        model.forward(dummy_input)
        for dummy_input in dummy_inputs
    ])

    print(f'latency (ms): {1e3 * (time() - t0) / 30}')

def basic_to_onnx(model:torch.nn.Module, onnx_path:str):
    torch.onnx.export(
    model, (torch.ones(1, 3, 192, 192)),
    onnx_path,
    opset_version=17,
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            # Regular addition is not supported for quantized operands. Hence, this way
            # ref. https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv2.py
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

class QuantizedModel(torch.nn.Module):
    def __init__(self, model_fp32):
        
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # FP32 model
        self.model_fp32 = model_fp32
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def static_quantize_mobilenetv2():
    untr_model = MobileNetV2(num_classes=960, block=QuantizableInvertedResidual)

    print("base model")
    print(untr_model)
    untr_model.to('cpu').eval()

    untr_model_copy = copy.deepcopy(untr_model)
    quantized_untr_model = QuantizedModel(untr_model_copy)
    quantized_untr_model.to('cpu').eval()
    quantized_untr_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model_fp32_fused = quantized_untr_model

    # #fuse layers
    # TODO: Requires in-depth walkthrough of sub-modules to fuse layers at each level
    # ref. https://leimao.github.io/blog/PyTorch-Static-Quantization/
    # model_fp32_fused = torch.quantization.fuse_modules(
    #     quantized_untr_model, ['features.0.0', 'features.0.1'] # TODO: check layer names
    # )

    prepared_model = torch.quantization.prepare(model_fp32_fused)
    img_size = 192

    prepared_model(torch.randn(1, 3, img_size, img_size))

    quantized_model = torch.quantization.convert(prepared_model)
    quantized_model.to('cpu').eval()
    print("quantized model")
    print(quantized_model)
    out = quantized_model(torch.randn(1, 3, img_size, img_size))
    print(out.shape)
    calculate_latency(quantized_model, 192)
    print('calculating time taken')
    avg, std = get_time_elapsed(quantized_model, 192, onnx=False)
    print(f'avg {avg} std {std}')

def walkthrough_model(model:torch.nn.Module, module_list, parent = ''):
    for module_name, module in model.named_children():
        full_name = f'{parent}/{module_name}'
        module_list.append((full_name, module))
        # print(f'module name: {full_name}')
        # print(f'type {type(module)}')
        # if('torch.nn.quantized.modules.conv.Conv2d' in str(type(module))):
        #     if (hasattr(module, 'weight')):
        #         breakpoint()
        #         for w in module.weight:
        #             print(f'w {w}')
        #         print(f'weights {module.weight}')
        walkthrough_model(module, module_list, full_name)

def copy_parameters(model:torch.nn.Module, quantized_model:torch.nn.Module):
    modules = []
    quantized_modules = []
    walkthrough_model(model, modules)
    walkthrough_model(quantized_model, quantized_modules)
    print(f'modules {len(modules)}')
    print(f'qmodules {len(quantized_modules)}')

    # Skip through initial quant modules to find first Sequential module
    # to match with non-quantized modules list
    qm_idx = 0
    while(True):
        _, qm = quantized_modules[qm_idx]
        if (type(qm) == torch.nn.modules.container.Sequential):
            break
        else:
            qm_idx += 1

    print(f'qmidx {qm_idx}')
    
    m_idx = 0
    while(m_idx < len(modules)):
        #working_copy.features._modules['0']._modules['0'].weight.data = 
        # quantized_model.model_fp32.features._modules['0']._modules['0'].weight().dequantize()
        if (type(modules[m_idx][1]) == torch.nn.modules.conv.Conv2d or
        type(modules[m_idx][1]) == torch.nn.modules.linear.Linear):
            # print(f'copying weights from {quantized_modules[qm_idx][0]} to {modules[m_idx][0]}')
            modules[m_idx][1].weight.data = quantized_modules[qm_idx][1].weight().dequantize()

        if (type(modules[m_idx][1]) == torch.nn.modules.batchnorm.BatchNorm2d):
            # print(f'copying bias from {quantized_modules[qm_idx][0]} to {modules[m_idx][0]}')
            modules[m_idx][1].bias.data = quantized_modules[qm_idx][1].bias.dequantize()

        m_idx += 1
        qm_idx += 1

def quantize_model(saved_state:Path) -> torch.nn.Module:
    model = tvqntmodels.mobilenet_v2(pretrained=False, num_classes = 960)
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend='qnnpack')
    # model.fuse_model()
    torch.quantization.prepare_qat(model, inplace=True)
    model.load_state_dict(torch.load(saved_state, map_location='cpu'))
    converted_model = torch.quantization.convert(model, inplace=False)
    converted_model.to('cpu').eval()
    return converted_model

def perform_qat(model:torch.nn.Module, quantized_model_path:str, quantized_onnx_path:str, dummy_training:bool = False) -> torch.nn.Module:
    if model is None:
        model = tvqntmodels.mobilenet_v2(pretrained=False, num_classes = 960)
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend='qnnpack')
    # model.fuse_model()
    torch.quantization.prepare_qat(model, inplace=True)

    #Perform training
    if dummy_training:
        img_size = 192
        model(torch.randn(1, 3, img_size, img_size))
    else:
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # To overcome 'imgcodecs: OpenEXR codec is disabled' error
        args, *_ = LitLandmarksTrainer.parse_args()
        LitLandmarksTrainer.train(args, model=model)

    torch.save(model.state_dict(), quantized_model_path)
    converted_model = torch.quantization.convert(model, inplace=False)
    converted_model.to('cpu').eval()
    # torch.save(m, quantized_model_path, _use_new_zipfile_serialization = True)
    calculate_latency(converted_model, 192)
    print(get_time_elapsed(converted_model, 192, onnx=False))

    # print(converted_model(torch.ones(1, 3, 192, 192)))
    # test_torch_model('', converted_model)
    # print('testing complete')
    # print('testing with path')
    # test_torch_model(quantized_model_path)
    
    # to_onnx(converted_model, quantized_onnx_path, (192, 192))
    # basic_to_onnx(converted_model, quantized_onnx_path)
    return converted_model


#%%
def mobilenetv2_qat():
#     return
    untr_model = MobileNetV2(num_classes=960, block=QuantizableInvertedResidual)

    print("base model")
    #print(untr_model)
    untr_model.to('cpu').eval()
    calculate_latency(untr_model, 192)
    torch.save(untr_model, '/home/yrajas/tmp/qat/untr_model.pt')

    untr_model_copy = copy.deepcopy(untr_model)

    print("walkthrough untr_model")
    # walkthrough_model(untr_model)

    quantized_untr_model = QuantizedModel(untr_model_copy)
    quantized_untr_model.eval()

        # elif backend == 'qnnpack':
        # model.qconfig = torch.quantization.QConfig(  # type: ignore[assignment]
        #     activation=torch.quantization.default_observer,
        #     weight=torch.quantization.default_weight_observer)
    quantized_untr_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model_fp32_fused = quantized_untr_model

    # #fuse layers
    # TODO: Requires in-depth walkthrough of sub-modules to fuse layers at each level
    # ref. https://leimao.github.io/blog/PyTorch-Static-Quantization/
    # model_fp32_fused = torch.quantization.fuse_modules(
    #     quantized_untr_model, ['features.0.0', 'features.0.1'] # TODO: check layer names
    # )

    model_fp32_fused.train()
    prepared_model = torch.quantization.prepare_qat(model_fp32_fused)
    prepared_model.train()
    img_size = 192
    prepared_model(torch.randn(1, 3, img_size, img_size))

    #Perform QAT
    # os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # To overcome 'imgcodecs: OpenEXR codec is disabled' error
    # args, *_ = LitLandmarksTrainer.parse_args()
    # LitLandmarksTrainer.train(args, prepared_model)

    print("before conversion")
    #print(prepared_model)

    quantized_model = torch.quantization.convert(prepared_model)
    quantized_model.to('cpu').eval()
    print("quantized model")
    print("walkthrough quantized_model")
    # walkthrough_model(quantized_model)

    #working_copy.features._modules['0']._modules['0'].weight.data = quantized_model.model_fp32.features._modules['0']._modules['0'].weight().dequantize()
    #print(quantized_model)
    #out = quantized_model(torch.randn(1, 3, img_size, img_size))

    # print(out.shape)
    # calculate_latency(quantized_model, 192)
    print('calculating time taken')
    # copy_parameters(untr_model, quantized_model)
    torch.save(quantized_model, '/home/yrajas/tmp/qat/quantized_model.pt')
    # untr_model(torch.randn(1, 3, img_size, img_size))
    # to_onnx(untr_model, '/home/yrajas/tmp/qatcopyparams/mobilenetv2_qat_copyparams.onnx', (img_size, img_size))
    # avg, std = get_time_elapsed(quantized_model, 192, onnx=True)
    # print(f'avg {avg} std {std}')

#%%
def get_test_dataset(identities:list, frames:int = 0):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # To overcome 'imgcodecs: OpenEXR codec is disabled' error
    data_module = SyntheticsDataModule(
        batch_size=5, 
        num_workers=1, 
        validation_proportion=0.1, 
        landmarks_definition='dense_320',
        roi_landmarks_definition='dense_320',
        landmarks_weights='dense_320_weights',
        roi_size=(192, 192),
        roi_size_multiplier=1.1,
        use_sigma=True,
        warp_affine=True,
        warp_scale=0.05,
        warp_rotate=10,
        warp_shift=0.05,
        warp_jiggle=0.05,
        warp_squash_chance=0.0,
        motion_blur_chance=0.05,
        data_dir='/home/yrajas/data/groundtruth_render_20220419_155805/', 
        frames_per_identity=frames, 
        identities=identities, 
        preload=True,
        load_depth=False,
        load_seg=False)
    data_module.setup()
    return data_module.full_dataset

def test_model(onnx_model_path:str):
    # MR model real data evaluation
    width = 192
    height = 192
    synthetics = get_test_dataset(list(range(0,20000,156)), 3)
    label_coords_unnormalized = torch.stack([s.ldmks_2d for s in synthetics])
    input_img = np.array([s.bgr_img.numpy() for s in synthetics])

    onnx_session = InferenceSession(str(onnx_model_path))
    input_name = onnx_session.get_inputs()[0].name
    predicted_coords_normalized = onnx_session.run([], {input_name: input_img})[0]

    predicted_coords_normalized = torch.tensor(predicted_coords_normalized)[:,:640] # ignore sigma
    predicted_coords_normalized = predicted_coords_normalized.reshape(-1, 320, 2) # reshape as co-ordinates
    predicted_coords_unnormalized = unnormalize_coordinates(predicted_coords_normalized, width, height)
    error = landmarks_error(predicted_coords_unnormalized, label_coords_unnormalized)

    print(f"Error [Val] {error.mean()}")

def test_torch_model(model_path:str, model:torch.nn.Module = None):
    # MR model real data evaluation
    width = 192
    height = 192
    synthetics = get_test_dataset(list(range(0,20000,156)), 3)
    label_coords_unnormalized = torch.stack([s.ldmks_2d for s in synthetics])
    input_img = torch.stack([s.bgr_img for s in synthetics])

    if model is None:
        model = torch.load(model_path)

    model.to('cpu').eval()
    with torch.no_grad():
        predicted_coords_normalized = model(input_img)
        predicted_coords_normalized = torch.tensor(predicted_coords_normalized)[:,:640] # ignore sigma
        predicted_coords_normalized = predicted_coords_normalized.reshape(-1, 320, 2) # reshape as co-ordinates
        predicted_coords_unnormalized = unnormalize_coordinates(predicted_coords_normalized, width, height)
        error = landmarks_error(predicted_coords_unnormalized, label_coords_unnormalized)

        print(f"Error [Val] {error.mean()}")

def static_quantize_onnx_model(onnx_model_path:str, quantized_onnx_model_path:str):
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
    class DummyReader(CalibrationDataReader):
        def __init__(self) -> None:
            super().__init__()
            self.dataset = get_test_dataset(list(range(int(20000))))
            self.length = len(self.dataset)
            self.idx = 0

        def get_next(self) -> dict:
            if self.idx < self.length:
                img = np.array([self.dataset[self.idx].bgr_img.numpy()])
                data = {'input': img}
                self.idx += 1
                return data
            else:
                return None
    # class RandomReader(CalibrationDataReader):
    #     def __init__(self) -> None:
    #         super().__init__()
    #         self.idx = 0

    #     def get_next(self) -> dict:
    #         img_size = 192
    #         if self.idx < 10:
    #             self.idx += 1
    #             return {'input': torch.randn(1, 3, img_size, img_size).numpy()}
    #         else:
    #             return None
    quantize_static(onnx_model_path, quantized_onnx_model_path, calibration_data_reader=DummyReader())
    
def deepspeedcompress_qat():
    config_path = '/home/yrajas/vision.hu.face.synthetics.training/face_synthetics_training/training/landmarks2/nas/dscompress.json'
    model = MobileNetV2(num_classes=960)
    # model = torch.load('/home/yrajas/tmp/qat/tvbaseline/captured_output_dense_320_mobilenetv2_100_192.pt')

    model_copy = copy.deepcopy(model)
    model_copy.to('cpu').eval()
    print("***before***")
    calculate_latency(model_copy, 192)
    # for name, module in model.named_modules():
    #     print(name)
    # print(model)
    # for p in model.parameters():
    #     print(p.norm())

    model = init_compression(model=model, deepspeed_config=config_path)
    model, _, _, _ = deepspeed.initialize(model=model, config=config_path)
    model.train()

    # for _ in range(100):
    #     _ = model(torch.randn(1, 3, 192, 192).cuda())
    #Perform QAT
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # To overcome 'imgcodecs: OpenEXR codec is disabled' error
    args, *_ = LitLandmarksTrainer.parse_args()
    LitLandmarksTrainer.train(args, model)

    #model = redundancy_clean(model=model, deepspeed_config=config_path)

    print("****after**")
    # for p in model.parameters():
    #     print(p.norm())

    model.module.to('cpu').eval()
    calculate_latency(model.module, 192)
    
    # torch.onnx.export(
    # model, (torch.randn(1, 3, 192, 192)),
    # '~/tmp/test.onnx',
    # opset_version=11,
    # verbose=False,
    # input_names=['input_0'],
    # output_names=['output_0'])
    print("done")

def train_mobilenetv2():
    model = MobileNetV2(num_classes=960)
    model.train()
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # To overcome 'imgcodecs: OpenEXR codec is disabled' error
    args, *_ = LitLandmarksTrainer.parse_args()
    LitLandmarksTrainer.train(args, model)

def train_torchvisionmodel():
    model = tvqntmodels.mobilenet_v2(pretrained=False, num_classes = 960)
    torch.save(model, '/home/yrajas/tmp/qat/tvbaseline/before_training.pt')
    model.train()
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1" # To overcome 'imgcodecs: OpenEXR codec is disabled' error
    args, *_ = LitLandmarksTrainer.parse_args()
    LitLandmarksTrainer.train(args, model)
    torch.save(model, '/home/yrajas/tmp/qat/tvbaseline/after_training.pt')

def convert_to_onnx(model_path:str, onnx_path:str, img_size:int):
    model = torch.load(model_path)

    to_onnx(model, onnx_path, (img_size, img_size))

#%%
def main():
    # calculate_latency(torch.load('/home/yrajas/tmp/qat_nightly/tv_noqat_baseline/outputs/dense_320_mobilenetv2_100_192.pt'), 192)
    # calculate_latency(torch.load('/home/yrajas/tmp/qat_nightly/tv_qat_mrpretrained/outputs/dense_320_mobilenetv2_100_192.pt'), 192)

    # get_1cpu_latency('/home/yrajas/tmp/qat_nightly/tv_noqat_baseline/outputs/dense_320_mobilenetv2_100_192.onnx', onnx=True)
    # get_1cpu_latency('/home/yrajas/tmp/qat_nightly/tv_qat_mrpretrained/outputs/dense_320_mobilenetv2_100_192.onnx', onnx=True)
    # print(profile_onnx('/home/yrajas/tmp/qat_nightly/tv_noqat_baseline/outputs/dense_320_mobilenetv2_100_192.onnx', img_size=(192, 192)))
    # print(get_time_elapsed(torch.load('/home/yrajas/tmp/qat_nightly/tv_noqat_baseline/outputs/dense_320_mobilenetv2_100_192.pt'), img_size=192, onnx=True))

    # test_torch_model('/home/yrajas/tmp/qat_nightly/tv_noqat_baseline/outputs/dense_320_mobilenetv2_100_192.pt')
    # test_torch_model('/home/yrajas/tmp/qat_nightly/tv_qat_mrpretrained/outputs/dense_320_mobilenetv2_100_192.pt')

    test_model('/home/yrajas/tmp/qat_nightly/tv_noqat_baseline/outputs/dense_320_mobilenetv2_100_192.onnx')
    test_model('/home/yrajas/tmp/qat_nightly/tv_qat_mrpretrained/outputs/dense_320_mobilenetv2_100_192.onnx')

    # qat_model = quantize_model('/home/yrajas/tmp/qat/tv_qat_lr1e-4/tv_qat_lr1e-4.pt')
    # test_torch_model(model_path='', model = qat_model)

    # torchvision_qat()
    # train_torchvisionmodel()
    # model = perform_qat(
    #     torch.load('/home/yrajas/tmp/qat/tvbaseline/captured_output_dense_320_mobilenetv2_100_192.pt'), 
    #     '/home/yrajas/tmp/qat/tv_qat_swa/tv_qat_swa.pt', 
    #     '/home/yrajas/tmp/qat/test_qat.onnx',
    #     dummy_training = False)
    # test_torch_model(model_path='', model=model)
    # print('exporting to onnx')
    # basic_to_onnx(model, '/home/yrajas/tmp/qat/tvbaseline/captured_output_dense_320_mobilenetv2_100_192_test_qat.onnx')
    # train_mobilenetv2()
    #static_quantize_mobilenetv2()

    # test_torch_model('/home/yrajas/tmp/qat/tvqat_mrtorch1_10/dense_320_mobilenetv2_100_192_mrtorch110_qat.pt')
    # test_torch_model('/home/yrajas/tmp/qat/tvbaseline/captured_output_dense_320_mobilenetv2_100_192.pt')
    # test_torch_model('/home/yrajas/tmp/qat/tvbaseline/captured_output_dense_320_mobilenetv2_100_192_qat.pt')
    #mobilenetv2_qat()

    # convert_to_onnx('/home/yrajas/tmp/qat/quantized_model.pt', '/home/yrajas/tmp/qat/quantized_model.onnx', 192)

    # deepspeedcompress_qat()

    # calculate_latency(torch.load('/home/yrajas/tmp/qat/baseline/outputs/dense_320_mobilenetv2_100_192.pt'), 192)
    
    # test_model('/home/yrajas/tmp/qat/quantized_static_model.onnx')
    # static_quantize_onnx_model(
    #     onnx_model_path = '/home/yrajas/tmp/qat/quantized_model.onnx',
    #     quantized_onnx_model_path='/home/yrajas/tmp/qat/quantized_static_model.onnx'
    #     )

    # test_model('/home/yrajas/tmp/qat/quantized_static_model.onnx')
    # test_torch_model('/home/yrajas/tmp/qat/baseline/outputs/dense_320_mobilenetv2_100_192.pt')
    # static_quantize_onnx_model(
    #     onnx_model_path = '/home/yrajas/tmp/mr20k/outputs/dense_320_mobilenetv2_100_192.onnx',
    #     quantized_onnx_model_path='/home/yrajas/tmp/mr20k/outputs/dense_320_mobilenetv2_100_192_randcalib_stqntz.onnx'
    #     )

    # test_torch_model('/home/yrajas/tmp/qat/quantized_model.pt')
    # print(get_model_flops(torch.load('/home/yrajas/tmp/qat/quantized_model.pt'), 192))

    # print(profile_onnx('/home/yrajas/tmp/qat/quantized_static_model.onnx', (192, 192)))

    # test model error for timm and quantized from timm
    # test_model('/home/yrajas/tmp/mr20k/outputs/dense_320_mobilenetv2_100_192.onnx')
    # test_model('/home/yrajas/tmp/mr20k/outputs/dense_320_mobilenetv2_100_192_static_quantized.onnx')

    # test model error for timm and random calibrated quantized from timm
    # test_model('/home/yrajas/tmp/mr20k/outputs/dense_320_mobilenetv2_100_192.onnx')
    # test_model('/home/yrajas/tmp/mr20k/outputs/dense_320_mobilenetv2_100_192_randcalib_stqntz.onnx')

    # test model error for orig paper based mbv2 and quantized form of it
    # test_model('/home/yrajas/tmp/mbv2paper/mbnetv2_paper.onnx')
    # test_model('/home/yrajas/tmp/mbv2paper/mbnetv2_paper_stquantized.onnx')

if __name__ == '__main__':
    main()