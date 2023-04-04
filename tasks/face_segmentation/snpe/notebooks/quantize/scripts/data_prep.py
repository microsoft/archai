import os
import argparse
import logging
import mlflow
import zipfile
from create_data import create_dataset


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data (zip file)")
    parser.add_argument("--output", type=str, help="path to resulting quantization data")
    args = parser.parse_args()

    input_path = args.data
    data_dir = 'data'
    output_dir = args.output

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", input_path)
    print("input output:", output_dir)

    if not input_path or not os.path.exists(input_path):
        raise Exception(f'### Error: no .zip file found in the {input_path} folder')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Start Logging
    mlflow.start_run()

    print('unzipping the data...')
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print('Converting the .png images to SNPE quantization .bin files...')
    create_dataset(data_dir, output_dir, 'quant', [256, 256, 3], 1000)

    for name in os.listdir(output_dir):
        print(name)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()