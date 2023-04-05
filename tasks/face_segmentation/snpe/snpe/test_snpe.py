# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
from datetime import datetime
from pathlib import Path
import csv
import os
import sys
import time
import tqdm
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
sys.path += [os.path.join(SCRIPT_DIR, '..', 'util')]
from shell import Shell
sys.path += [os.path.join(SCRIPT_DIR, '..', 'vision')]
from collect_metrics import get_metrics
from shutil import rmtree, copyfile
from onnxruntime import InferenceSession

TASK = os.path.basename(os.getcwd())
MODEL = "model"
TEST_IMAGES = os.path.join('data', 'test')
VERBOSE = False
MAX_BATCH_SIZE = 1000
DEVICE = None

# device parameters
# the /data mount on the device has 64GB available.
DEVICE_WORKING_DIR = "/data/local/tmp"
RANDOM_INPUTS = 'random_inputs'
RANDOM_INPUT_LIST = 'random_raw_list.txt'
SNPE_ROOT = None
INPUT_LIST_FILENAME = 'input_list_for_device.txt'
snpe_target_arch = None


def set_device(device):
    global DEVICE
    DEVICE = device
    shell = Shell()
    shell.run(os.getcwd(), adb("root"))


def get_device():
    global DEVICE
    if DEVICE:
        return DEVICE
    else:
        raise Exception("Please specify the '--device' to use")


def _get_input_layout(shape):
    # snpe-onnx-to-dlc supported input layouts are:
    # NCHW, NHWC, NFC, NCF, NTF, TNF, NF, NC, F, NONTRIVIAL
    # N = Batch, C = Channels, H = Height, W = Width, F = Feature, T = Time
    if shape[0] == 3:
        # then the RGB channel is first, so we are NCHW
        return 'NCHW'
    elif shape[-1] == 3:
        return 'NHWC'
    else:
        raise Exception(f"Cannot figure out input layout from shape: {shape}")


def onnx_to_dlc(model, model_dir):
    sess = InferenceSession(model, providers=['CPUExecutionProvider'])
    if len(sess._sess.inputs_meta) > 1:
        raise Exception("Cannot handle models with more than one input")
    if len(sess._sess.outputs_meta) > 1:
        raise Exception("Cannot handle models more than one output")

    input_meta = sess._sess.inputs_meta[0]
    output_meta = sess._sess.outputs_meta[0]

    filename = os.path.basename(model)
    basename = os.path.splitext(filename)[0]

    print(f"==> Converting model {model} to .dlc...")

    shape = input_meta.shape
    if len(shape) == 4:
        shape = shape[1:]  # trim off batch dimension
    layout = _get_input_layout(shape)
    input_shape = ",".join([str(i) for i in input_meta.shape])
    output_dlc = f"{model_dir}/{basename}.dlc"
    # snpe-onnx-to-dlc changes the model input to NHWC.
    dlc_shape = shape if layout == 'NHWC' else [shape[1], shape[2], shape[0]]

    command = "snpe-onnx-to-dlc " + \
        f"-i \"{model}\" " + \
        f"-d {input_meta.name} \"{input_shape}\" " + \
        f"--input_layout {input_meta.name} {layout} " + \
        f"--out_node \"{output_meta.name}\" " + \
        f"-o \"{output_dlc}\" "

    print(command)

    shell = Shell()
    rc = shell.run(os.getcwd(), command, True)
    if 'Conversion completed successfully' not in rc:
        raise Exception(rc)

    # the DLC model undoes the NCHW and results in a .dlc model that takes NHWC input.
    return [output_dlc, dlc_shape]


def create_snpe_config(model_file, output_dir):
    sess = InferenceSession(model_file, providers=['CPUExecutionProvider'])
    if len(sess._sess.inputs_meta) > 1:
        raise Exception("Cannot handle models with more than one input")
    if len(sess._sess.outputs_meta) > 1:
        raise Exception("Cannot handle models more than one output")

    input_meta = sess._sess.inputs_meta[0]
    output_meta = sess._sess.outputs_meta[0]

    shape = input_meta.shape
    if len(shape) == 4:
        shape = shape[1:]  # trim off batch dimension
    layout = _get_input_layout(shape)
    input_shape = ",".join([str(i) for i in input_meta.shape])
    # snpe-onnx-to-dlc changes the model input to NHWC
    output_shape = shape if layout == 'NHWC' else [shape[1], shape[2], shape[0]]

    config = {
        "io_config": {
            "input_names": [input_meta.name],
            "input_shapes": [input_meta.shape],
            "output_names": [output_meta.name],
            "output_shapes": [output_shape]
        },
        "convert_options": {
            "input_layouts": [layout]
        },
        "quantize_options": {
            "use_enhanced_quantizer": True
        }
    }

    config_file = os.path.join(output_dir, 'snpe_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    return config


def convert_onnx_to_dlc(onnx_model, snpe_output_file):
    from olive.passes import SNPEConversion
    config = create_snpe_config(onnx_model.model_path, os.path.dirname(snpe_output_file))
    convert_options = config['io_config']
    convert_options["input_layouts"] = config['convert_options']["input_layouts"]

    snpe_conversion = SNPEConversion(convert_options, disable_search=True,)
    snpe_model = snpe_conversion.run(onnx_model, snpe_output_file)

    assert Path(snpe_model.model_path).is_file()

    dlc_model_path = snpe_model.model_path
    # output_shapes is a list, but we only want the first element
    dlc_shape = snpe_model.io_config["output_shapes"][0]

    return dlc_model_path, dlc_shape


def convert_model(model, model_dir):
    """ Converts the given model from .onnx form to .dlc and returns
    the path to the .dlc file, the input shape, and an optional error message if something went wrong."""
    if not os.path.isfile(model):
        print(f"model to convert not found in {model}")
        sys.exit(1)

    # make sure we run the downloaded model and not some old model
    # left over from a previous run.
    if os.path.isdir(model_dir):
        rmtree(model_dir)
    os.makedirs(model_dir)

    from olive.model import ONNXModel

    filename = os.path.basename(model)
    basename, ext = os.path.splitext(filename)

    if ext != ".onnx":
        print("convert_model was not provided with a valid .onnx model")
        sys.exit(1)

    try:
        onnx_model = ONNXModel(model, basename)
        snpe_output_file = f"{model_dir}/model.dlc"
        model, shape = convert_onnx_to_dlc(onnx_model, snpe_output_file)
        return [model, shape, None]
    except Exception as ex:
        return [None, None, str(ex)]


def create_quant_dataloader(data_dir):
    from olive.snpe import SNPEProcessedDataLoader
    return SNPEProcessedDataLoader(data_dir, input_list_file="input_list.txt")


def quantize_model(model, onnx_model, snpe_model_dir):
    """ Returns tuple containing the quantized model file and optional error message """
    from olive.model import SNPEModel
    from olive.passes import SNPEQuantization

    snpe_model_dir = os.path.realpath(snpe_model_dir)  # Olive requires the full path
    basename = os.path.splitext(os.path.basename(model))[0]
    output_model = os.path.join(snpe_model_dir, f"{basename}.quant.dlc")
    full_dlc_path = os.path.join(snpe_model_dir, f"{basename}.dlc")

    data_dir = os.path.join('data', 'quant')

    config = create_snpe_config(onnx_model, snpe_model_dir)
    if config is None:
        return [None, "### SNPE Configuration file could not be loaded"]

    snpe_model = SNPEModel(model_path=full_dlc_path, name=basename, **config["io_config"])

    quant_options = {
        "use_enhanced_quantizer": True,
        "data_dir": data_dir,
        "dataloader_func": create_quant_dataloader
    }

    # tbd: what is "enable_htp": True
    snpe_quantization = SNPEQuantization(quant_options, disable_search=True)

    try:
        snpe_quantized_model = snpe_quantization.run(snpe_model, output_model)
    except Exception as ex:
        error = None
        for line in str(ex):
            if '[ERROR]' in line:
                error = line
        if not error:
            error = str(ex)
        return [None, error]

    if not Path(snpe_quantized_model.model_path).is_file():
        return [None, "### Model conversion failed"]

    return [snpe_quantized_model.model_path, None]


def adb(cmd):
    return f"adb -s {get_device()} {cmd}"


def download_results(input_images, start, output_dir):
    shell = Shell()
    result = shell.run(os.getcwd(), adb(f"shell ls {DEVICE_WORKING_DIR}/{TASK}/{MODEL}/output"), False)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Now download results from output folder full of 'Result_nnn' folders, copying the file to
    # down to match with the input image name, so the input image name might be 000000.bin, and
    # the downloaded file might be named 'Result_0/logits_0:1.raw' and this will become
    # the downloaded file named `000000.raw` in the 'snpe_output' folder.  This makes it easier
    # to match the the output with the input later when we are analyzing the results.
    output_filename = None
    index = start
    for name in input_images:
        line = f"Result_{index}"
        if line in result:
            raw_file = os.path.splitext(name)[0] + '.raw'
            index += 1
            if not output_filename:
                cmd = adb(f"shell ls {DEVICE_WORKING_DIR}/{TASK}/{MODEL}/output/{line}/")
                device_result = shell.run(os.getcwd(), cmd, False)
                output_filename = device_result.strip()
            print(f"{line}/{output_filename} ===> {raw_file}")
            device_result = f"{DEVICE_WORKING_DIR}/{TASK}/{MODEL}/output/{line}/{output_filename}"
            rc = shell.run(output_dir, adb(f"pull {device_result} {raw_file}"), False)
            if "error:" in rc:
                print("### error downloading results: " + rc)
                sys.exit(1)


def get_target_arch(snpe_root):
    global SNPE_ROOT
    SNPE_ROOT = snpe_root
    if not os.path.isdir(snpe_root):
        print("SNPE_ROOT folder {} not found".format(snpe_root))
        sys.exit(1)
    for name in os.listdir(os.path.join(snpe_root, 'lib')):
        if name.startswith('aarch64-android'):
            print(f"Using SNPE_TARGET_ARCH {name}")
            return name

    print("SNPE_ROOT folder {} missing aarch64-android-*".format(snpe_root))
    sys.exit(1)


def clear_images():
    shell = Shell()
    target = f"{DEVICE_WORKING_DIR}/{TASK}/data/test"
    shell.run(os.getcwd(), adb(f"shell \"rm -rf {target}\""))


def copy_file(shell, folder, filename, target):
    rc = shell.run(folder, adb(f"push {filename} {target}"), False)
    if "error:" in rc:
        print(f"### Error copying file {filename}: {rc}")
        sys.exit(1)


def setup_images(folder, batch):
    shell = Shell()
    target = f"{DEVICE_WORKING_DIR}/{TASK}/data/test"
    shell.run(os.getcwd(), adb(f"shell \"mkdir -p {target}\""))
    # since we are doing a batch we have to generate the input_list_for_device.txt file.
    list_file = INPUT_LIST_FILENAME
    with open(list_file, 'w', encoding='utf-8') as f:
        for line in batch:
            f.write(f"{target}/{line}\n")
    copy_file(shell, os.getcwd(), list_file, target)

    # pushing the whole dir often fails for some reason, so we push individual files with a retry loop
    with tqdm.tqdm(total=len(batch)) as pbar:
        for file in batch:
            pbar.update(1)
            retries = 5
            while retries > 0:
                try:
                    copy_file(shell, folder, file, target)
                    break
                except Exception as e:
                    retries -= 1
                    print(f"Error {e}, retrying in 1 second ...")
                    time.sleep(1)
                    if retries == 0:
                        raise Exception(f"Cannot copy file {file} to {target}")


def get_setup():
    global snpe_target_arch
    if not snpe_target_arch:
        print(f"snpe_target_arch is not set")
        sys.exit(1)

    lib_path = f"{DEVICE_WORKING_DIR}/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
    setup = f"export SNPE_TARGET_ARCH={snpe_target_arch} && " + \
        f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{DEVICE_WORKING_DIR}/{snpe_target_arch}/lib && " + \
        f"export PATH=$PATH:{DEVICE_WORKING_DIR}/{snpe_target_arch}/bin && " + \
        f"export ADSP_LIBRARY_PATH='{lib_path}' && " + \
        f"cd {DEVICE_WORKING_DIR}/{TASK}/{MODEL}"

    print("Using environment:")
    print(setup)
    return setup


def upload_first_image(dataset):
    target = f"{DEVICE_WORKING_DIR}/{TASK}/data/test"
    shell = Shell()
    shell.run(os.getcwd(), adb(f"shell \"mkdir -p {target}\""))
    files = os.listdir(dataset)
    files.sort()
    filename = files[0]
    print(f"Uploading image {filename}...")
    shell.run(dataset, adb(f"push {filename} {target}"), VERBOSE)
    return os.path.join(target, filename)


def generate_random_inputs(count, shape):
    input_list_path = os.path.join(RANDOM_INPUTS, RANDOM_INPUT_LIST)
    meta = os.path.join(RANDOM_INPUTS, 'meta.json')
    if os.path.isfile(meta):
        with open(meta, 'r', encoding='utf-8') as f:
            info = json.load(f)
            if info['count'] == count and info['shape'] == str(shape):
                # then we can reuse these random inputs.
                return input_list_path
    if os.path.isdir(RANDOM_INPUTS):
        rmtree(RANDOM_INPUTS)
    os.makedirs(RANDOM_INPUTS)
    with open(meta, 'w', encoding='utf-8') as f:
        info = {'count': count, 'shape': str(shape)}
        json.dump(info, f, indent=2)
    random_data_dim = np.product(shape)
    with open(input_list_path, 'w', encoding='utf-8') as input_list:
        for i in range(count):
            rand_raw = np.random.uniform(-1.0, +1.0, random_data_dim).astype(np.float32)
            raw_filepath = os.path.join(RANDOM_INPUTS, 'random_input_' + str(i) + '.raw')
            input_list.write(raw_filepath + "\n")
            with open(raw_filepath, 'wb') as fid:
                fid.write(rand_raw)
    return input_list_path


def get_memlog_usage(benchmark_dir):
    # find the memory usage
    # Uptime: 1738234338 Realtime: 1738234338
    for i in range(1, 6):
        memlog = os.path.join(benchmark_dir, 'results', 'latest_results', 'mem', 'DSP_ub_tf8', f'Run{i}', 'MemLog.txt')
        if os.path.isfile(memlog):
            try:
                for line in open(memlog, 'r', encoding='utf-8').readlines():
                    if 'Realtime:' in line:
                        parts = line.strip().split(' ')
                        try:
                            mem = int(parts[-1])
                            if mem != 0:
                                return [mem, memlog]
                        except:
                            pass
            except:
                pass
    return [0, '']


def read_total_inference_avg(csvfile):
    col_index = 11
    with open(csvfile, 'r', encoding='utf-8') as file:
        for data in csv.reader(file):
            for i in range(len(data)):
                col = data[i]
                if col.startswith('DSP_ub_tf8_timing'):
                    col_index = i
                    break
            if 'Total Inference Time' in data:
                return int(data[col_index])
    return 0


def run_throughput(model, duration):
    if not model:
        print("### --run needs the --model parameter")
        sys.exit(1)
    shell = Shell()

    use_dsp = "--use_dsp" if 'quant' in model else ''
    setup = get_setup()
    dataset = os.path.join('data', 'test')
    filename = upload_first_image(dataset)

    print(f"Running throughput test for {duration} seconds...")
    rc = shell.run(
        os.getcwd(),
        adb(f"shell \"{setup} &&" +
            f"snpe-throughput-net-run --container ./{model} {use_dsp} --duration {duration} " +
            "--iterations 1 --perf_profile high_performance " +
            f"--input_raw {filename} \""), VERBOSE)
    lines = rc.split('\n')
    for out in lines:
        if "Total throughput:" in out:
            print(out)


def compute_results(shape, output_folder):
    if not os.path.isdir(output_folder):
        print("Folder not found: '{output_folder}'")
        return
    dataset = os.getenv("INPUT_DATASET")
    if not os.path.isdir(dataset):
        print("Please set your INPUT_DATASET environment variable")
        return

    return get_metrics(shape, False, dataset, output_folder)


def run_batches(onnx_model, dlc_model, images_dir, workspace_dir):

    from olive.snpe import (
        SNPESessionOptions,
        SNPEInferenceSession,
        SNPEProcessedDataLoader
    )

    input_dir = os.path.realpath(images_dir)
    snpe_model_dir = os.path.dirname(dlc_model)

    output_dir = os.path.join(workspace_dir, 'snpe-output')
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    full_dlc_path = os.path.realpath(dlc_model)
    basename = os.path.splitext(os.path.basename(dlc_model))[0]

    options = SNPESessionOptions(
        android_target = get_device(),
        device = "dsp" if 'quant' in basename else "cpu",
        workspace = workspace_dir,
        accumulate_outputs = True
    )

    config = create_snpe_config(onnx_model, snpe_model_dir)
    if config is None:
        return [None, "### SNPE Configuration file could not be loaded"]

    io_config = config["io_config"]

    # More than 1000 test images can fill up the device and then we run out of memory.
    # So we run the inference session in batches here.
    data_loader = SNPEProcessedDataLoader(input_dir, input_list_file='input_list.txt', batch_size=100)

    output_folder = ''
    latencies = []
    for i in range(data_loader.num_batches):
        print(f"Running SNPE inference batch {i} of {data_loader.num_batches}")
        batch_dir, batch_input_list, _ = data_loader.get_batch(i)
        session = SNPEInferenceSession(full_dlc_path, io_config, options)
        results = session.net_run(batch_input_list, batch_dir)
        output_folder = results['output_dir']
        latencies += [results['latencies']]

    return output_folder, latencies


def run_benchmark(onnx_model, dlc_model, images_dir, duration, workspace_dir):

    from olive.snpe import (
        SNPESessionOptions,
        SNPEInferenceSession,
        SNPEProcessedDataLoader
    )

    input_dir = os.path.realpath(images_dir)
    input_list = os.path.join(input_dir, 'input_list.txt')
    snpe_model_dir = os.path.dirname(dlc_model)

    output_dir = os.path.join(workspace_dir, 'snpe-output')
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    full_dlc_path = os.path.realpath(dlc_model)
    basename = os.path.splitext(os.path.basename(dlc_model))[0]

    # This mirrors what the snpe_bench.py script is doing.
    options = SNPESessionOptions(
        android_target=get_device(),
        device="dsp" if 'quant' in basename else "cpu",
        extra_args="--perf_profile high_performance --profiling_level basic",
        workspace=workspace_dir,
        accumulate_outputs=True
    )

    config = create_snpe_config(onnx_model, snpe_model_dir)
    if config is None:
        return [None, "### SNPE Configuration file could not be loaded"]

    data_loader = SNPEProcessedDataLoader(images_dir, input_list_file='input_list.txt', batch_size=50)
    io_config = config["io_config"]
    output_folder = ''
    latencies = []

    print("Running SNPE inference benchmark")
    batch_dir, batch_input_list, _ = data_loader.get_batch(0)
    session = SNPEInferenceSession(full_dlc_path, io_config, options)
    results = session.net_run(batch_input_list, batch_dir)
    output_folder = results['output_dir']
    latencies += [results['latencies']]

    return (output_folder, latencies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a model on the QUALCOMM DSP using adb and SNPE SDK to quantize ' +
                                     'the model')
    parser.add_argument('--device', '-d', help='The Android device id (as returned by adb devices)', default=None)
    parser.add_argument('--images', '-i', help='Location of local test image dataset (created with create_data.py)')
    parser.add_argument('--model', '-m', help='The path to the ONNX model to test')
    parser.add_argument('--dlc', help='The specific .dlc model to test, if not provided it will be converted from --model and stored in an snpe_models folder')
    parser.add_argument('--quantize', '-q', help='Quantize the given onnx or dlc model', action="store_true")
    parser.add_argument('--benchmark', '-b', help='Run a benchmark on the given model', action="store_true")
    parser.add_argument('--throughput', '-t', help='Run performance test of the given model', action="store_true")
    parser.add_argument('--duration', type=int, help='Duration of throughput and benchmark tests (default 10 seconds)', default=10)
    parser.add_argument('--verbose', '-v', help='Show all output (default false)', action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose
    if args.device:
        set_device(args.device)
    model = args.model
    MODEL_DIR = "snpe_models"

    if not args.model:
        print("Please provide an onnx model as input")
        sys.exit(1)

    ndk = os.getenv("ANDROID_NDK_ROOT")
    if not ndk:
        print("you must have a ANDROID_NDK_ROOT installed, see the readme.md")
        sys.exit(1)

    if args.dlc:
        dlc_model = args.dlc
    else:
        dlc_model, shape, error = convert_model(args.model, MODEL_DIR)
        if error:
            print(error)
            sys.exit(1)

    snpe = os.getenv("SNPE_ANDROID_ROOT")
    if not snpe:
        print("please set your SNPE_ANDROID_ROOT environment variable, see readme.md")
        sys.exit(1)

    snpe = os.getenv("SNPE_ROOT")
    if not snpe:
        print("please set your SNPE_ROOT environment variable, see readme.md")
        sys.exit(1)

    config = create_snpe_config(args.model, '.')
    shape = config['io_config']['output_shapes'][0]

    if args.quantize:
        quantized_model, error = quantize_model(model, model, MODEL_DIR)
        if error is not None:
            print(error)
            sys.exit(1)

    if args.throughput:
        run_throughput(model, args.duration)
        sys.exit(0)

    if args.benchmark:
        start = datetime.now()
        image_path = os.path.realpath(args.images)
        output_folder, latencies = run_benchmark(model, dlc_model, image_path, args.duration, '.')
        end = datetime.now()
        print(f"benchmark completed in {end-start} seconds, results in {output_folder}")
        for m in latencies:
            print(f"total_inference_time={m['total_inference_time']}")
        sys.exit(0)

    if args.images:
        start = datetime.now()
        output_folder, latencies = run_batches(model, dlc_model, args.images, '.')
        end = datetime.now()
        print(f"batch completed in {end-start} seconds, results in {output_folder}")
        for m in latencies:
            print(f"total_inference_time={m['total_inference_time']}")
        input_shape = (1, shape[0], shape[1], 19)
        compute_results(input_shape, output_folder)
