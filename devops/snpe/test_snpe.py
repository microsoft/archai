# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
from datetime import datetime
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
SNPE_TARGET_STL = "libgnustl_shared.so"
SNPE_BENCH = None
RANDOM_INPUTS = 'random_inputs'
RANDOM_INPUT_LIST = 'random_raw_list.txt'
SNPE_ROOT = None
snpe_target_arch = None

def set_device(device):
    global DEVICE
    DEVICE = device
    shell = Shell()
    shell.run(os.getcwd(), adb("root"))


def get_device():
    global DEVICE
    return DEVICE


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


def add_snpe_env(command):
    env = f"export PYTHONPATH={SNPE_ROOT}/lib/python && " + \
          f"export PATH=$PATH:{SNPE_ROOT}/bin/x86_64-linux-clang && "
    return env + command


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
    rc = shell.run(os.getcwd(), add_snpe_env(command), True)
    if 'Conversion completed successfully' not in rc:
        raise Exception(rc)

    # the DLC model undoes the NCHW and results in a .dlc model that takes NHWC input.
    return [output_dlc, dlc_shape]


def convert_model(model, model_dir, input_shape=None):
    """ Converts the given model from .onnx form to .dlc and returns
    the path to the .dlc file, True if a conversion was run, and an optional
    error message if something went wrong.  Also returns the input shape.
    It also returns True if the model is already quantized.  So the result
    has 4 parts [model_file, shape, already_quantized, optional_error_message] """
    if not os.path.isfile(model):
        print(f"model to convert not found in {model}")
        sys.exit(1)

    # make sure we run the downloaded model and not some old model
    # left over from a previous run.
    if os.path.isdir(model_dir):
        rmtree(model_dir)
    os.makedirs(model_dir)

    ext = os.path.splitext(model)[1]
    if ext == ".onnx":
        try:
            model, shape = onnx_to_dlc(model, model_dir)
        except Exception as ex:
            return [None, None, False, str(ex)]
    elif ext == '.dlc':
        # have to assume the shape in this case
        if not input_shape:
            input_shape = [256, 256, 3]
        if '.quant' in model:
            # already quantized, has to be in the model_dir
            if os.path.dirname(model) == model_dir:
                return [model, input_shape, True, None]
            else:
                newfile = os.path.join(model_dir, os.path.basename(model))
                if model != newfile:
                    copyfile(model, newfile)
                return [newfile, input_shape, True, None]

    if model is None:
        return [None, None, False, f"### Model extension {ext} not supported"]

    return [model, shape, False, None]


def quantize_model(model, model_dir):
    """ Returns tuple containing the quantized model file and optional error message """
    quant_set = os.path.join('data', 'quant', 'input_list.txt')
    if not os.path.isfile(quant_set):
        print(f"Quantize dataset {quant_set} is missing")
        sys.exit(1)

    basename = os.path.splitext(os.path.basename(model))[0]
    output_model = os.path.join(model_dir, f"{basename}.quant.dlc")
    full_dlc = os.path.join(model_dir, f"{basename}.dlc")

    command = "snpe-dlc-quantize " + \
        f"--input_dlc \"{full_dlc}\" " + \
        f"--input_list \"{quant_set}\" " + \
        f"--output_dlc \"{output_model}\" " + \
        "--use_enhanced_quantizer"

    print(f"==> Quantizing model {model}...(this can take several minutes)...")
    print(command)
    shell = Shell()
    try:
        shell.run(os.getcwd(), add_snpe_env(command), True)
    except Exception as ex:
        error = None
        for line in str(ex):
            if '[ERROR]' in line:
                error = line
        if not error:
            error = str(ex)
        return [None, error]

    if not os.path.isfile(output_model):
        return [None, "### Model conversion failed"]

    return [output_model, None]


def adb(cmd):
    if DEVICE:
        return f"adb -s {DEVICE} {cmd}"
    else:
        print("Please specify the --device to use")
        sys.exit(1)


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


def setup_libs(snpe_root):
    global snpe_target_arch
    snpe_target_arch = get_target_arch(snpe_root)

    print("Pushing SNPE binaries and libraries to device...")
    shell = Shell()
    for dir in [f"{DEVICE_WORKING_DIR}/{snpe_target_arch}/bin",
                f"{DEVICE_WORKING_DIR}/{snpe_target_arch}/lib",
                f"{DEVICE_WORKING_DIR}/dsp/lib"]:
        shell.run(os.getcwd(), adb(f"shell \"mkdir -p {dir}\""))

    shell.run(
        os.path.join(snpe_root, "lib", snpe_target_arch),
        adb(f"push . {DEVICE_WORKING_DIR}/{snpe_target_arch}/lib"), VERBOSE)

    shell.run(
        os.path.join(snpe_root, "lib", 'dsp'),
        adb(f"push . {DEVICE_WORKING_DIR}/dsp/lib"), VERBOSE)

    for program in ['snpe-net-run', 'snpe-parallel-run', 'snpe-platform-validator', 'snpe-throughput-net-run']:
        shell.run(
            os.path.join(snpe_root, "bin", snpe_target_arch),
            adb(f"push {program} {DEVICE_WORKING_DIR}/{snpe_target_arch}/bin"), VERBOSE)

        shell.run(
            os.path.join(snpe_root, "bin", snpe_target_arch),
            adb(f"shell \"chmod u+x {DEVICE_WORKING_DIR}/{snpe_target_arch}/bin/{program}\""))


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
    list_file = 'input_list_for_device.txt'
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


def setup_model(model):
    print(f"copy {model} file to device...")
    shell = Shell()
    target = f"{DEVICE_WORKING_DIR}/{TASK}/{MODEL}"
    shell.run(os.getcwd(), adb(f"shell \"rm -rf {target}\""), False)
    shell.run(os.getcwd(), adb(f"shell \"mkdir -p {target}\""), False)
    dir = os.path.dirname(model)
    filename = os.path.basename(model)
    if not dir:
        dir = "snpe_models"
        model = os.path.join(dir, filename)
    if not os.path.isfile(model):
        print(f"setup_model: model not found: {model}")
        sys.exit(1)
    shell.run(dir, adb(f"push {filename} {target}"), VERBOSE)
    return filename


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


def run_test(model):
    if not model:
        print("### --run needs the --model parameter")
        sys.exit(1)

    global snpe_target_arch
    if not snpe_target_arch:
        print(f"snpe_target_arch is not set")
        sys.exit(1)

    shell = Shell()
    # make sure any previous run output is cleared.
    shell.run(os.getcwd(), adb(f'shell \"rm -rf  {DEVICE_WORKING_DIR}/{TASK}/{MODEL}/output\"'))

    use_dsp = "--use_dsp" if 'quant' in model else ''
    setup = get_setup()
    shell.run(
        os.getcwd(),
        adb(f"shell \"export SNPE_TARGET_ARCH={snpe_target_arch} && {setup} &&" +
            f"snpe-net-run --container ./{model} --input_list ../data/test/input_list_for_device.txt {use_dsp}\""))


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


def run_benchmark(model, name, shape, snpe_root, iterations, random_input_count):
    global SNPE_BENCH
    if not model:
        print("Must provide a --model file")
        sys.exit(1)

    if not snpe_root:
        print("Please provide the --snpe root dir")
        sys.exit(1)
    elif not os.path.isdir(snpe_root):
        print(f"The --snpe {snpe_root} not found")
        sys.exit(1)

    global snpe_target_arch
    snpe_target_arch = get_target_arch(snpe_root)

    cwd = os.getcwd()
    benchmark_dir = os.path.join(cwd, name, 'benchmark')
    if os.path.isdir(benchmark_dir):
        rmtree(benchmark_dir)
    os.makedirs(benchmark_dir)

    model_path = os.path.join(cwd, model)

    files_list_path = generate_random_inputs(random_input_count, shape)

    if not DEVICE:
        print("Please specify the --device to use")
        sys.exit(1)

    config = {
        "Name": "model",
        "HostRootPath": benchmark_dir,
        "HostResultsDir": os.path.join(benchmark_dir, "results"),
        "HostName": "localhost",
        "DevicePath": os.path.join(DEVICE_WORKING_DIR, "snpebm"),
        "Devices": [DEVICE],
        "Runs": iterations,
        "Model": {
            "Name": "model",
            "Dlc": model_path,
            "InputList": files_list_path,
            "Data": [os.path.dirname(files_list_path)]
        },
        "Runtimes": ["DSP"],
        "Measurements": ['timing'],
        "CpuFallback": False,
        "PerfProfile": "high_performance",
        "ProfilingLevel": "detailed",
        "BufferTypes": ["ub_tf8"]
    }

    config_file = os.path.join(benchmark_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    if SNPE_BENCH is None:
        from snpe_bench import snpe_bench   # noqa: F811
        SNPE_BENCH = snpe_bench

    SNPE_BENCH('snpe_bench', ['-c', config_file, '-s', '2'])

    # find the memory usage
    csvfile = os.path.join(benchmark_dir, 'results', 'latest_results', 'benchmark_stats_model.csv')
    total_inference_avg = read_total_inference_avg(csvfile)

    print(f"Total inference avg={total_inference_avg}")
    return [total_inference_avg, csvfile]


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


def compute_results(shape):
    image_size = tuple(shape)[1:2]
    output_dir = os.path.join('snpe_output')
    if not os.path.isdir(output_dir):
        print("No 'snpe_output' folder, please run --download first")
        return
    dataset = os.getenv("INPUT_DATASET")
    if not os.path.isdir(dataset):
        print("Please set your INPUT_DATASET environment variable")
        return

    return get_metrics(image_size, False, dataset, output_dir)


def run_batches(model, snpe_root, images, output_dir):

    global snpe_target_arch
    snpe_target_arch = get_target_arch(snpe_root)

    files = [x for x in os.listdir(images) if x.endswith(".bin")]
    files.sort()

    if os.path.isdir(output_dir):
        rmtree(output_dir)

    print("Found {} input files".format(len(files)))
    start = 0
    while start < len(files):
        if start + MAX_BATCH_SIZE < len(files):
            print(f"==================== Running Batch of {MAX_BATCH_SIZE} from {start} ==============================")
        clear_images()
        batch = files[start: start + MAX_BATCH_SIZE]
        setup_images(images, batch)
        run_test(model)
        download_results(batch, start, output_dir)
        start += MAX_BATCH_SIZE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a model on the QUALCOMM DSP using adb and SNPE SDK to quantize ' +
                                     'the model')
    parser.add_argument('--snpe', '-s', help='Location of SNPE SDK for setup on the device (defaults to SNPE_ROOT)')
    parser.add_argument('--device', '-d', help='The Android device id (as returned by adb devices)', default=None)
    parser.add_argument('--images', '-i', help='Location of local test image dataset (created with create_data.py)',
                        default=TEST_IMAGES)
    parser.add_argument('--model', '-m', help='Name of model to test (e.g. model.quant.dlc).  If you name a .onnx ' +
                        'model it will convert it to .dlc and quantize it.')
    parser.add_argument('--benchmark', '-b', help='Run snpe_benchmark on the given model', action="store_true")
    parser.add_argument('--iterations', type=int, help='Number of benchmark iterations (default 5)', default=5)
    parser.add_argument('--throughput', '-t', help='Run performance test of the given model', action="store_true")
    parser.add_argument('--duration', type=int, help='Duration of throughput test (default 10 seconds)', default=10)
    parser.add_argument('--verbose', '-v', help='Show all output (default false)', action="store_true")
    args = parser.parse_args()

    VERBOSE = args.verbose
    set_device(args.device)
    model = args.model
    MODEL_DIR = "snpe_models"
    OUTPUT_DIR = "snpe_output"

    ndk = os.getenv("ANDROID_NDK_ROOT")
    if not ndk:
        print("you must have a ANDROID_NDK_ROOT installed, see the readme.md")
        sys.exit(1)

    model, shape, quantized, error = convert_model(args.model, MODEL_DIR)
    if error:
        print(error)
        sys.exit(1)

    if not quantized:
        model, error = quantize_model(model, MODEL_DIR)
        if error:
            print(error)
            sys.exit(1)

    snpe = args.snpe
    if not snpe:
        snpe = os.getenv("SNPE_ROOT")
        if not snpe:
            print("please set your SNPE_ROOT environment variable, see readme.md")
            sys.exit(1)

    snpe_target_arch = get_target_arch(snpe)

    if snpe:
        sys.path += [f'{snpe}/benchmarks', f'{snpe}/lib/python']
        from snpe_bench import snpe_bench  # noqa: F401
        SNPE_IMPORTED = True
        setup_libs(snpe)

    if model:
        model = setup_model(model)

    if args.throughput:
        run_throughput(model, args.duration)
        sys.exit(0)

    if args.benchmark:
        start = datetime.now()
        run_benchmark(os.path.join(MODEL_DIR, model), '.', shape, snpe, args.iterations, 50)
        end = datetime.now()
        print(f"benchmark completed in {end-start} seconds")
        sys.exit(0)

    if args.images:
        run_batches(model, snpe, args.images, OUTPUT_DIR)
        compute_results(shape)
