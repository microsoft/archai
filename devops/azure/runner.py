# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# This script monitors an Azure blob store using connection string defined
# in the environmentvariable "FaceSyntheticsModelStorageAccount".
# When a file named "go" shows up in a folder containing a new model to test
# it downloads the model, runs the F1 score check with 10,000 images
# and the snpe_bench benchmark and posts the 2 results .csv files
# back to the same folder.  The .csv files are named:
# - test_results.csv - for the F1 score.
# - benchmark_stats_model.csv - for benchmark perf numbers.
import argparse
import json
import os
import sys
import time
import platform
import statistics
import tracemalloc
import gc
import psutil
import logging
import traceback
from shutil import rmtree
from status import get_status, get_all_status_entities, get_utc_date, merge_status_entity
from usage import add_usage
from azure.data.tables import EntityProperty, EdmType

# This file contains wrappers on the snpe execution of the model connecting everything
# to the Azure table describing the jobs that need to be run, and keeping the status
# rows up to date while these jobs are running.


CONNECTION_NAME = 'MODEL_STORAGE_CONNECTION_STRING'
SNPE_OUTPUT_DIR = 'snpe_output'
MODEL_DIR = 'model'
SNPE_MODEL_DIR = 'snpe_models'
SNPE_OUTPUT_DIR = 'snpe_output'
MAX_BENCHMARK_RUNS = 5
BENCHMARK_INPUT_SIZE = 50
CLEAR_RANDOM_INPUTS = 0
LOG_FILE_NAME = 'memusage.log'
DEVICE_FILE = "device.txt"
UNIQUE_NODE_ID = None
BENCHMARK_RUN_COUNT = 0

SCRIPT_DIR = os.path.dirname(__file__)
sys.path += [os.path.join(SCRIPT_DIR, '..', 'snpe')]
sys.path += [os.path.join(SCRIPT_DIR, '..', 'util')]
sys.path += [os.path.join(SCRIPT_DIR, '..', 'vision')]
rss_start = None

# Set the logging level for all azure-* libraries
logging.getLogger('azure').setLevel(logging.ERROR)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)

from test_snpe import setup_libs, convert_model, quantize_model, setup_model, run_benchmark
from test_snpe import run_batches, set_device, get_device
from create_data import create_dataset
from collect_metrics import get_metrics
from priority_queue import PriorityQueue
from upload import upload_blob
from download import download_model, has_model
from test_onnx import test_onnx
from dlc_helper import view_model, get_dlc_metrics


logger = logging.getLogger(__name__)


def log(msg):
    print(msg)
    logger.info(msg)


def read_shape(dir):
    shape_file = os.path.join(dir, 'shape.txt')
    if os.path.isfile(shape_file):
        with open(shape_file, 'r', encoding='utf-8') as f:
            return eval(f.readline().strip())
    return [0, 0, 0]


def save_shape(dir, shape):
    shape_file = os.path.join(dir, 'shape.txt')
    with open(shape_file, 'w', encoding='utf-8') as f:
        f.write(str(shape))
    return [0, 0, 0]


def check_device(device, snpe_root):
    set_device(device)
    device_info = ''
    if os.path.isfile(DEVICE_FILE):
        with open(DEVICE_FILE, 'r', encoding='utf-8') as f:
            device_info = f.readline().strip()
    if device_info != device:
        with open(DEVICE_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{device}\n")
    # always setup libs in case device was changed by someone else.
    setup_libs(snpe_root)


def check_dataset(shape, name, test_size):
    test = os.path.join('data', name)
    if os.path.isdir(test):
        s = read_shape(test)
        if s != shape:
            log(f"recreating {name} folder since shape needs to change from {s} to {shape}")
            rmtree(test)
        else:
            bins = [x for x in os.listdir(test) if x.endswith('.bin')]
            if len(bins) != test_size:
                log(f"recreating test folder since it had {len(bins)} images")
                rmtree(test)

    if not os.path.isdir(test):
        create_dataset(dataset, name, shape, test_size)
        save_shape(test, shape)


def convert(name, entity, long_name, model_path):
    log("Converting model: " + long_name)
    entity['model_name'] = long_name
    entity['status'] = 'converting'
    merge_status_entity(entity)

    shape = None
    if 'shape' in entity:
        shape = eval(entity['shape'])

    model_dir = os.path.join(name, SNPE_MODEL_DIR)
    model, input_shape, quantized, error = convert_model(model_path, model_dir, shape)
    if error:
        entity['status'] = 'error'
        entity['error'] = error
        merge_status_entity(entity)
        return 'error'

    if shape != input_shape:
        entity['shape'] = str(input_shape)
        merge_status_entity(entity)

    if not quantized:
        log("Uploading converted model: " + model)
        upload_blob(name, model)

    return model


def quantize(name, entity, model):
    entity['status'] = 'quantizing'
    merge_status_entity(entity)

    input_shape = eval(entity['shape'])
    check_dataset(input_shape, 'quant', 1000)

    model_dir = os.path.join(name, SNPE_MODEL_DIR)
    model, error = quantize_model(model, model_dir)
    if error:
        entity['status'] = 'error'
        entity['error'] = error
        merge_status_entity(entity)
        return 'error'

    # save the quantized .dlc since it takes so long to produce.
    log("Uploading quantized model: " + model)
    upload_blob(name, model)
    return model


def get_unique_node_id():
    global UNIQUE_NODE_ID
    if UNIQUE_NODE_ID:
        return UNIQUE_NODE_ID
    return platform.node()


def set_unique_node_id(id):
    global UNIQUE_NODE_ID
    UNIQUE_NODE_ID = id


def is_locked(entity):
    node = get_unique_node_id()
    if 'node' in entity and entity['node']:
        name = entity['name']
        locked = entity['node']
        if locked != node:
            log(f"{node}: model {name} is running on: {locked}")
            return 'busy'
    return None


def lock_job(entity, service=None):
    node = get_unique_node_id()
    name = entity['name']
    # make sure we have the most up to date version of the entity.
    entity = get_status(name, service=service)
    retries = 10
    while retries:
        retries -= 1
        if is_locked(entity):
            # someone beat us to it
            raise Exception('lock encountered')
        entity['node'] = node
        try:
            merge_status_entity(entity, service=service)
            break
        except:
            # someone beat us to it!
            log("lock failed, entity changed by someone else, trying again...")

    # make sure we really got the lock!
    entity = get_status(name)
    if 'node' in entity and entity['node'] == node:
        return entity
    # someone beat us to it
    raise Exception('lock encountered')


def unlock_job(entity, service=None):
    node = get_unique_node_id()
    # make sure we have the most up to date version of the entity.
    entity = get_status(entity['name'], service=service)
    if 'node' in entity:
        if entity['node'] and entity['node'] != node:
            lock = entity['node']
            raise Exception(f'cannot unlock entity because it is locked by someone else ({lock})')
        else:
            entity['node'] = ''
            retries = 10
            while retries:
                retries -= 1
                try:
                    merge_status_entity(entity, service=service)
                    break
                except:
                    # someone beat us to it!
                    log("unlock failed, entity changed by someone else, trying again...")

    return entity


def run_onnx(name, dataset, model_path, test_size):
    out_dir = os.path.join(name, SNPE_OUTPUT_DIR, 'onnx_outputs')
    if os.path.isdir(out_dir):
        rmtree(out_dir)
    test_onnx(dataset, model_path, out_dir, test_size)
    return out_dir


def is_complete(entity, prop):
    return prop in entity


def is_true(entity, prop):
    return prop in entity and entity[prop]


def get_total_inference_avg(entity):
    if 'total_inference_avg' in entity and entity['total_inference_avg']:
        try:
            return json.loads(entity['total_inference_avg'])
        except:
            pass
    return []


def benchmarks_complete(entity):
    return len(get_total_inference_avg(entity))


def get_mean_benchmark(entity):
    avg = get_total_inference_avg(entity)
    if len(avg) > 0:
        return int(statistics.mean(avg))
    return 0


def get_int64_value(entity, name):
    x = 0
    if name in entity:
        x = entity[name]
        if isinstance(x, EntityProperty) and x.edm_type is EdmType.INT64:
            x = x.value
        else:
            x = int(x)
    return x


def benchmark(entity, model, name):
    global BENCHMARK_RUN_COUNT, CLEAR_RANDOM_INPUTS

    input_shape = eval(entity['shape'])
    # next highest priority is to get benchmark times
    total_benchmark_runs = benchmarks_complete(entity)

    if (total_benchmark_runs >= MAX_BENCHMARK_RUNS):
        return False  # nothing to do

    if total_benchmark_runs < MAX_BENCHMARK_RUNS:
        BENCHMARK_RUN_COUNT += 1
        if CLEAR_RANDOM_INPUTS > 0 and BENCHMARK_RUN_COUNT >= CLEAR_RANDOM_INPUTS:
            clear_random_inputs()
            BENCHMARK_RUN_COUNT = 0
        log(f"Running benchmark iteration {total_benchmark_runs} of {MAX_BENCHMARK_RUNS}...")
        entity['status'] = 'running benchmark'
        merge_status_entity(entity)

        start = get_utc_date()
        ifs, perf_result = run_benchmark(model, name, input_shape, snpe_root, 5, BENCHMARK_INPUT_SIZE)
        end = get_utc_date()
        add_usage(get_device(), start, end)

        upload_blob(name, perf_result)

        total_inference_avg = get_total_inference_avg(entity)
        total_inference_avg += [ifs]
        entity['total_inference_avg'] = json.dumps(total_inference_avg)
        mean = int(statistics.mean(total_inference_avg))
        entity['mean'] = mean
        if len(total_inference_avg) > 1:
            stdev = int(statistics.stdev(total_inference_avg))
            entity['stdev'] = int((stdev * 100) / mean)
        total_benchmark_runs += 1
    else:
        mean = get_mean_benchmark(entity)

    if is_benchmark_only(entity, False) and total_benchmark_runs == MAX_BENCHMARK_RUNS:
        entity['status'] = 'complete'
        entity['completed'] = get_utc_date()
    merge_status_entity(entity)
    return True


def run_model(name, snpe_root, dataset, conn_string, use_device, benchmark_only, no_quantization):
    log("===================================================================================================")
    log(f"Checking model: {name} on node {get_unique_node_id()}")
    log("===================================================================================================")

    with open('name.txt', 'w', encoding='utf-8') as file:
        file.write(name + '\n')

    # make sure we have a clean slate and don't pick up old files from previous runs
    model_dir = os.path.join(name, MODEL_DIR)
    if os.path.isdir(model_dir):
        rmtree(model_dir)
    os.makedirs(model_dir)
    snpe_model_dir = os.path.join(name, SNPE_MODEL_DIR)
    if os.path.isdir(snpe_model_dir):
        rmtree(snpe_model_dir)
    snpe_output_dir = os.path.join(name, SNPE_OUTPUT_DIR)
    if os.path.isdir(snpe_output_dir):
        rmtree(snpe_output_dir)
    benchmark_dir = os.path.join(name, 'benchmark')
    if os.path.isdir(benchmark_dir):
        rmtree(benchmark_dir)

    entity = get_status(name)

    onnx_model_found, long_name, onnx_model = download_model(name, model_dir, conn_string, no_dlc=True)
    if not onnx_model_found or not os.path.isfile(onnx_model):
        entity['status'] = 'error'
        entity['error'] = 'missing model'
        merge_status_entity(entity)
        log(f"### no model found for {name}")
        return

    # see if we have converted the model or not.
    # do this first no matter what.
    converted = has_model(name, conn_string, 'model.dlc')
    is_quantized = has_model(name, conn_string, 'model.quant.dlc')
    if not is_quantized:
        # oh, the quant model disappeared so clear the flag so it gets
        # quantized again by a machine that can do that.
        if 'quantized' in entity:
            del entity['quantized']
            merge_status_entity(entity)
        if no_quantization:
            return

    if not converted and onnx_model_found:
        model = convert(name, entity, long_name, onnx_model)
        if model == 'error':
            return
    elif converted:
        model_found, long_name, model = download_model(name, snpe_model_dir, conn_string, 'model.dlc')
        if not model_found:
            raise Exception('### internal error, the model.dlc download failed!')
    elif not is_quantized and not converted:
        entity['status'] = 'error'
        entity['error'] = 'missing model'
        merge_status_entity(entity)
        log(f"### no model found for {name}")
        return

    # see if we have a quantized model or not.
    model = os.path.join(snpe_model_dir, 'model.dlc')
    if not is_quantized:
        model = quantize(name, entity, model)
        if model == 'error':
            return
        entity['quantized'] = True
        if 'macs' in entity:
            del entity['macs']  # need to redo it since we re-quantized.
    else:
        entity['quantized'] = True

    quantized_model = os.path.join(snpe_model_dir, 'model.quant.dlc')
    if not os.path.isfile(quantized_model):
        _, long_name, model = download_model(name, snpe_model_dir, conn_string, 'model.quant.dlc')
        if not model or not os.path.isfile(model):
            raise Exception("??? quantized model should exist at this point...")
    else:
        model = quantized_model

    log(f"==> using model {model}")

    if 'macs' not in entity:
        html = view_model(model)
        macs, params = get_dlc_metrics(html)
        entity['macs'] = macs
        entity['params'] = EntityProperty(params, EdmType.INT64)
        entity['status'] = 'converted'
        merge_status_entity(entity)
        upload_blob(name, html)
        return

    if use_device:
        if benchmark(entity, model, name):
            return

    if benchmark_only:
        log(f"Benchmark only has nothing to do on model {name}")
        return

    # next highest priority is to get the 1k f1 score.
    test_size = 0
    prop = None
    if use_device and not is_complete(entity, 'f1_1k'):
        test_size = 1000
        prop = 'f1_1k'
    elif use_device and not is_complete(entity, 'f1_1k_f'):
        test_size = 1000
        prop = 'f1_1k_f'
        if not converted:
            # this is a model that is prequantized, we don't have the original
            entity[prop] = 'n/a'
            entity['status'] = '.dlc model not found'
            merge_status_entity(entity)
            return
        model = os.path.join(snpe_model_dir, 'model.dlc')
        os.remove(quantized_model)  # make sure we can't run this one.
    elif use_device and not is_complete(entity, 'f1_10k'):
        test_size = 10000
        prop = 'f1_10k'
    elif not is_complete(entity, 'f1_onnx'):
        test_size = 10000
        prop = 'f1_onnx'
        model = onnx_model
    else:
        # why are we here?
        return

    input_shape = eval(entity['shape'])
    input_size = tuple(input_shape)[0:2]  # e.g. (256,256)

    # copy model to the device.
    if prop != 'f1_onnx':
        # now that we have the shape, we can create the appropriate quant and test
        # datasets!
        check_dataset(input_shape, 'test', test_size)
        filename = setup_model(model)

    if prop == 'f1_onnx':
        if not onnx_model_found:
            entity[prop] = 'n/a'
            entity['status'] = 'onnx file not found'
            merge_status_entity(entity)
            return
        else:
            entity['status'] = f'Running {prop}'
            merge_status_entity(entity)
            snpe_output_dir = run_onnx(name, dataset, onnx_model, test_size)
    else:
        entity['status'] = f'Running {prop}'
        merge_status_entity(entity)
        test_input = os.path.join('data', 'test')
        start = get_utc_date()
        run_batches(filename, snpe_root, test_input, snpe_output_dir)
        end = get_utc_date()
        add_usage(get_device(), start, end)

    try:
        use_pillow = 'use_pillow' in entity and entity['use_pillow']
        test_results, chart, f1score = get_metrics(input_size, False, dataset, snpe_output_dir, use_pillow)
    except Exception as ex:
        entity['status'] = 'error'
        entity['error'] = str(ex)
        merge_status_entity(entity)
        return

    log(f"### Saving {prop} score of {f1score}")
    entity[prop] = f1score
    merge_status_entity(entity)
    upload_blob(name, test_results, f"test_results_{prop}.png")
    upload_blob(name, chart, f"pr_curve_{prop}.csv")

    if 'f1_1k' in entity and 'f1_10k' in entity and 'f1_1k_f' in entity and 'f1_onnx' in entity:
        entity['status'] = 'complete'
        log(f"Completed {name}")
        merge_status_entity(entity)


def clear_random_inputs():
    if os.path.isdir('random_inputs'):
        log("Clearing random_inputs.")
        rmtree('random_inputs')


def is_benchmark_only(entity, benchmark_only):
    benchmark_only_flag = benchmark_only
    if 'benchmark_only' in entity:
        benchmark_only_flag = get_int64_value(entity, 'benchmark_only')
    return benchmark_only_flag


def node_quantizing():
    """ Ee don't want to do more than one quantization at a time on a given node
    because it is an CPU intensive operation. """
    id = platform.node() + '_'
    count = 0
    for e in get_all_status_entities(status='complete', not_equal=True):
        status = ''
        if 'status' in e:
            status = e['status']
        if 'node' not in e:
            continue
        node = e['node']
        if node.startswith(id) and node != get_unique_node_id() and \
           (status == 'converting' or status == 'quantizing'):
            count += 1
    return count > 0


def find_work_prioritized(use_device, benchmark_only, subset_list, no_quantization):
    queue = PriorityQueue()
    quantizing = no_quantization or node_quantizing()
    for entity in get_all_status_entities(status='complete', not_equal=True):
        name = entity['name']
        if subset_list is not None and name not in subset_list:
            log(f"# skipping model {name} because it is in the subset list")
            continue
        total_benchmark_runs = benchmarks_complete(entity)
        if is_locked(entity):
            log(f"# skip entity {name} because someone else is working on it")
            continue
        if 'error' in entity:
            log(f"# skipping {name} because something went wrong on previous step.")
            continue
        if not is_complete(entity, 'macs') or not is_true(entity, 'quantized'):
            if quantizing:
                if no_quantization:
                    log(f"No quantization work is done on this node. Skip {name} for now until other node works on it.")
                else:
                    log(f"skip {name} for now until other quantization finishes on our node")
                continue
            priority = 20
        elif use_device and (total_benchmark_runs < MAX_BENCHMARK_RUNS):
            priority = 30 + total_benchmark_runs
        elif is_benchmark_only(entity, benchmark_only):
            continue
        elif not is_complete(entity, 'f1_onnx'):
            priority = 60
        elif use_device and not is_complete(entity, 'f1_1k'):
            priority = 100 + get_mean_benchmark(entity)
        elif use_device and not is_complete(entity, 'f1_1k_f'):
            priority = 100 + get_mean_benchmark(entity) * 10
        elif use_device and not is_complete(entity, 'f1_10k'):
            # prioritize by how fast the model is!
            priority = 100 + get_mean_benchmark(entity) * 100
        else:
            # this model is done!
            continue

        if 'priority' in entity:
            # allow user to override the priority
            priority = get_int64_value(entity, 'priority')

        queue.enqueue(priority, entity)
    return queue


def garbage_collect():
    # remove old folders so we don't grow disk usage forever
    now = time.time()
    one_day = 60 * 60 * 24
    for f in list(os.listdir()):
        if os.path.isdir(f) and f != 'data' and f != 'random_inputs' and f != 'DEVICE_FILE':
            mod = os.path.getmtime(f)
            if now - mod > one_day:
                log(f"Garbage collecting {f}...")
                rmtree(f)


def monitor(snpe_root, dataset, use_device, benchmark_only, subset_list, no_quantization):
    global rss_start
    conn_string = os.getenv(CONNECTION_NAME)
    if not conn_string:
        log(f"Please specify your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    logging.basicConfig(filename=LOG_FILE_NAME, filemode='a',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    rss = psutil.Process(os.getpid()).memory_info().rss
    if rss_start is None:
        rss_start = rss
    growth = rss / rss_start

    file_mod = os.path.getmtime(__file__)

    # terminate this script if the memory has grown too much or the script
    # itself has been modified.  This will cause the outer 'loop.sh' to
    # loop and start a fesh process and pick any code modifications.
    while growth < 10:
        if file_mod != os.path.getmtime(__file__):
            log("Code has changed, need to restart.")
            sys.exit(0)

        try:
            queue = find_work_prioritized(use_device, benchmark_only, subset_list, no_quantization)
        except Exception as e:
            log(f"Error in find_work_prioritized: {e}")
            time.sleep(60)
            continue

        if queue.size() == 0:
            log("No work found.")
            sys.exit(0)
        else:
            garbage_collect()

            # do the top priority job then go back to find_work_prioritized in case
            # other jobs were add/completed in parallel while this was executing.
            priority, entity = queue.dequeue()
            name = entity['name']
            try:
                entity = lock_job(entity)
                benchmark_only_flag = is_benchmark_only(entity, benchmark_only)
                gc.collect()
                tracemalloc.start()
                snapshot1 = tracemalloc.take_snapshot()
                run_model(name, snpe_root, dataset, conn_string, use_device, benchmark_only_flag, no_quantization)
                gc.collect()
                snapshot2 = tracemalloc.take_snapshot()
                logging.info(f"========= memory diff vms={rss} growth={growth}============")
                for i in snapshot2.compare_to(snapshot1, 'lineno')[:10]:
                    logging.info(i)

                unlock_job(entity)
            except Exception as e:
                errorType, value, stack = sys.exc_info()
                if str(e) == 'lock encountered':
                    log('model is running on another machine')
                elif 'ConnectionResetError' in str(e):
                    log('ConnectionResetError: Ignoring Azure flakiness...')
                    unlock_job(entity)
                else:
                    # bug in the script somewhere... don't leave the node locked.
                    log(f'### Exception: {errorType}: {value}')
                    for line in traceback.format_tb(stack):
                        log(line.strip())

                    unlock_job(entity)
                    sys.exit(1)

        time.sleep(10)  # give other machines a chance to grab work so we don't get stuck in retry loops.
        rss = psutil.Process(os.getpid()).memory_info().rss
        if rss_start is None:
            rss_start = rss
        growth = rss / rss_start

    # we terminate here to reclaim the leaked memory, and to ensure we shut down cleanly without
    # leaving any rows in the table locked, we have an outer loop.sh script that will restart the runner.
    log("Memory leak detected")
    sys.exit(0)


if __name__ == '__main__':
    snpe_root = os.getenv("SNPE_ROOT")
    if not snpe_root:
        log("Please specify your 'SNPE_ROOT' environment variable.")
        sys.exit(1)
    if not os.path.isdir(snpe_root):
        log(f"Your SNPE_ROOT '{snpe_root} is not found.")
        sys.exit(1)

    sys.path += [f'{snpe_root}/benchmarks', f'{snpe_root}/lib/python']
    # make sure we can import snpe_bench!
    from snpe_bench import snpe_bench  # noqa: F401

    ndk = os.getenv("ANDROID_NDK_ROOT")
    if not ndk:
        log("you must have a ANDROID_NDK_ROOT installed, see the ../device/readme.md")
        sys.exit(1)
    if not os.path.isdir(ndk):
        log(f"Your ANDROID_NDK_ROOT '{ndk} is not found.")
        sys.exit(1)

    dataset = os.getenv("INPUT_DATASET")
    if not dataset:
        log("please provide --input or set your INPUT_DATASET environment variable")
        sys.exit(1)
    if not os.path.isdir(dataset):
        log(f"Your INPUT_DATASET '{dataset} is not found.")
        sys.exit(1)

    con_str = os.getenv(CONNECTION_NAME)
    if not con_str:
        log(f"Please set your {CONNECTION_NAME} environment variable.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Test the models as they appears in our Azure table')
    parser.add_argument('--device', '-d', help='Specify which Qualcomm device device to use (default None).')
    parser.add_argument('--benchmark', help='Run benchmark tests only (no F1 tests).', action="store_true")
    parser.add_argument('--max_benchmark_runs', type=int, help='Set maximum number of benchmark runs per model ' +
                        '(default 5).', default=5)
    parser.add_argument('--subset', help='Comma separated list of friendly model names to focus on, ' +
                        'ignoring all other models.')
    parser.add_argument('--clear_random_inputs', type=int, help='How many benchmark runs before clearing ' +
                        'random_inputs (default 0 means no clearing).', default=0)
    parser.add_argument('--no_quantization', help='Do not do any quantization work on this machine.',
                        action="store_true")
    parser.add_argument('--working', help='Use this working folder for all downloaded models and temp files ' +
                        '(default cwd).')

    args = parser.parse_args()
    if args.working:
        log(f"Using working folder: {args.working}")
        os.chdir(args.working)

    logger.setLevel('INFO')
    logger.addHandler(logging.FileHandler('runner.log', 'a'))

    MAX_BENCHMARK_RUNS = args.max_benchmark_runs
    CLEAR_RANDOM_INPUTS = args.clear_random_inputs
    device = args.device
    if device:
        set_unique_node_id(f"{platform.node()}_{device}")
        check_device(device, snpe_root)
    else:
        set_unique_node_id(platform.node())

    subset = None
    if args.subset:
        subset = [x.strip() for x in args.subset.split(',')]

    monitor(snpe_root, dataset, device is not None, args.benchmark, subset, args.no_quantization)
