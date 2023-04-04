import os
import sys
import argparse
import mlflow
from utils import spawn


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to quantization dataset folder")
    parser.add_argument("--model", type=str, help="path to model we need to quantize")
    parser.add_argument("--list_file", type=str, help="the name of the input_list.txt file")
    parser.add_argument("--output", type=str, help="place to write the quantized model")
    args = parser.parse_args()

    input_path = args.data
    list_file = args.list_file
    model_path = args.model
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir == '':
        output_dir = '.'

    print("input data:", input_path)
    print("input model:", model_path)
    print("output model:", output_path)

    if not input_path or not os.path.exists(input_path):
        raise Exception(f'### Error: no input data found at: {input_path}')

    if not input_path or not os.path.exists(model_path):
        raise Exception(f'### Error: no input model found at: {model_path}')

    if not output_path or not os.path.exists(output_dir):
        raise Exception(f'### Error: no output path found at: {output_dir}')

    os.makedirs(output_dir, exist_ok=True)

    # Start Logging
    mlflow.start_run()
    input_list = os.path.join(input_path, list_file)

    # snpe-dlc-quant needs full paths in the input list.
    with open(input_list, 'r') as f:
        lines = f.readlines()

    lines = [os.path.join(input_path, line) for line in lines]
    input_list = 'input_list.txt'  # not in the read-only input_path folder.
    with open(input_list, 'w') as f:
        f.writelines(lines)

    rc, stdout, stderr = spawn(['snpe-dlc-quant', '--input_dlc', model_path,
                               '--input_list', input_list,
                               '--output_dlc', output_path,
                               '--use_enhanced_quantizer'])

    print("stdout:")
    print("-------")
    print(stdout)

    print("")
    print("stderr:")
    print("-------")
    print(stderr)

    if "[INFO] Saved quantized dlc" in stderr:
        print("==> Quantization successful!")
    else:
        print("==> Quantization failed!")
        sys.exit(1)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
