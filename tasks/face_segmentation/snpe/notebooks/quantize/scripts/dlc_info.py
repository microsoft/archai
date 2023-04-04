import os
import sys
import argparse
import mlflow
from utils import spawn

def macs_to_float(macs):
    if macs.endswith('B'):
        return float(macs[:-1]) * 1e9
    if macs.endswith('M'):
        return float(macs[:-1]) * 1e6
    elif macs.endswith('K'):
        return float(macs[:-1]) * 1e3
    else:
        return float(macs)


def main():

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to dlc model we need to get info about")
    parser.add_argument("--output", type=str, help="the output text file to write to")
    args = parser.parse_args()

    model_path = args.model
    output_path = args.output

    print("input model:", model_path)
    print("output path:", output_path)
    print("tracking url:", mlflow.tracking.get_tracking_uri())

    if not model_path or not os.path.exists(model_path):
        raise Exception(f'### Error: no input model found at: {model_path}')

    # Start Logging
    mlflow.start_run()

    rc, stdout, stderr = spawn(['snpe-dlc-info', '--input_dlc', model_path])

    print("stdout:")
    print("-------")
    print(stdout)

    print("")
    print("stderr:")
    print("-------")
    print(stderr)

    with open(output_path, 'w') as f:
        f.write(stdout)

    params_prefix = 'Total parameters'
    macs_prefix = 'Total MACs per inference'
    memory_prefix = 'Est. Steady-State Memory Needed to Run:'

    # Parse stdout to get the info we want to log as metrics
    for line in stdout.split('\n'):
        if line.startswith(params_prefix):
            params = line.split(':')[1].split('(')[0].strip()
            mlflow.log_metric(params_prefix, float(params))
        elif line.startswith(macs_prefix):
            macs = line.split(':')[1].split('(')[0].strip()
            mlflow.log_metric(macs_prefix, macs_to_float(macs))
        elif line.startswith(memory_prefix):
            mem = line.split(':')[1].strip().split(' ')[0].strip()
            mlflow.log_metric('Estimated Memory Needed to Run', float(mem))

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
