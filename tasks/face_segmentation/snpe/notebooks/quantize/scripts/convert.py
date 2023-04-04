import os
import sys
import mlflow
import argparse
from onnxruntime import InferenceSession
from utils import spawn


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


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to onnx file to convert to .dlc")
    parser.add_argument("--output", type=str, help="path to resulting converted model")
    args = parser.parse_args()

    model = args.model
    output_path = args.output
    output_dir = os.path.dirname(output_path)

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input model:", model)
    print("output:", output_path)

    if not model or not os.path.exists(model):
        raise Exception(f'### Error: no file found at: {model}')

    if output_dir == '':
        output_dir = '.'
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Start Logging
    mlflow.start_run()

    print(f"Found mlflow tracking Uri: {mlflow.tracking.get_tracking_uri()}")

    sess = InferenceSession(model, providers=['CPUExecutionProvider'])
    if len(sess._sess.inputs_meta) > 1:
        raise Exception("Cannot handle models with more than one input")
    if len(sess._sess.outputs_meta) > 1:
        raise Exception("Cannot handle models more than one output")

    input_meta = sess._sess.inputs_meta[0]
    output_meta = sess._sess.outputs_meta[0]

    print(f"==> Converting model {model} to .dlc...")

    shape = input_meta.shape
    if len(shape) == 4:
        shape = shape[1:]  # trim off batch dimension
    layout = _get_input_layout(shape)
    input_shape = ",".join([str(i) for i in input_meta.shape])
    output_dlc = output_path
    # snpe-onnx-to-dlc changes the model input to NHWC.
    dlc_shape = shape if layout == 'NHWC' else [shape[1], shape[2], shape[0]]

    mlflow.set_tag("input_shape", dlc_shape)

    command = ["snpe-onnx-to-dlc", "-i",  model,
               "-d", input_meta.name, input_shape,
               "--input_layout", input_meta.name, layout,
               "--out_node", output_meta.name,
               "-o", output_dlc]

    print(" ".join(command))

    rc, stdout, stderr = spawn(command)

    print("shape: ", dlc_shape)

    print("stdout:")
    print("-------")
    print(stdout)

    print("")
    print("stderr:")
    print("-------")
    print(stderr)

    if "INFO_CONVERSION_SUCCESS" in stderr:
        print("==> Conversion successful!")
    else:
        print("==> Conversion failed!")
        sys.exit(1)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
