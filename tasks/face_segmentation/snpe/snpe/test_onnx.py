# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
from onnxruntime import InferenceSession, get_available_providers
import os
import numpy as np
import cv2
import sys
import tqdm
from create_data import DataGenerator


def test_onnx(dataset_dir, model, out_dir, test_size=1000, show=False):
    os.makedirs(out_dir, exist_ok=True)
    provider_list = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in get_available_providers():
        print("using gpu")
        provider_list = ['CUDAExecutionProvider'] + provider_list
    sess = InferenceSession(model, providers=provider_list)
    if len(sess._sess.inputs_meta) > 1:
        raise Exception("Cannot handle models with more than one input")
    if len(sess._sess.outputs_meta) > 1:
        raise Exception("Cannot handle models more than one output")

    input_meta = sess._sess.inputs_meta[0]
    output_meta = sess._sess.outputs_meta[0]
    shape = output_meta.shape
    if len(shape) == 4:
        shape = shape[1:]  # remove match dimension.

    oc, ow, oh = shape
    if oh < 20:
        ow, oh, oc = input_meta.shape

    shape = input_meta.shape
    print(f"input shape: {shape}")
    print(f"output shape: {shape}")
    if len(shape) == 4:
        shape = shape[1:]  # remove match dimension.
    w, h, c = shape
    transpose = (0, 1, 2)
    reverse = (0, 1, 2)
    if shape[0] == 3:
        # then we need to transpose the input.
        print("transposing to move RGB channel")
        transpose = (2, 0, 1)
        reverse = (1, 2, 0)
        c, w, h = shape
    input_name = input_meta.name

    data_gen = DataGenerator(dataset_dir, (w, h), subset='test', count=test_size, transpose=transpose)
    with tqdm.tqdm(total=len(data_gen)) as pbar:
        for fname, img in data_gen():
            inf = sess.run(None, {input_name: img[None, ...]})[0]
            inf = inf.reshape(inf.shape[1:])  # remove batch dimension
            inf = inf.transpose(reverse).reshape((ow, oh, -1))
            basename = os.path.splitext(os.path.basename(fname))[0]
            filename = os.path.join(out_dir, basename + ".raw")
            inf.tofile(filename)

            if show:
                # debug visualize
                img = img.transpose(reverse)
                cls_seg = np.argmax(inf, axis=-1)
                img = (255 * img).astype(np.uint8)
                norm = cv2.normalize(cls_seg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                cls_seg_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                canvas = np.concatenate([img[..., ::-1], cls_seg_color], axis=1)
                cv2.imshow('img', canvas)
                key = cv2.waitKey() & 0xFF
                if key == 27:
                    break

            pbar.update(1)


if __name__ == '__main__':
    model = os.path.join('model', 'model.onnx')
    output = os.path.join('onnx_outputs')

    parser = argparse.ArgumentParser(description='Run an ONNX model test on a batch of input images and write ' +
                                     'the outputs to a given folder')
    parser.add_argument('--input', help='Location of the original input images ' +
                        '(default INPUT_DATASET environment variable')
    parser.add_argument('--model', '-m', help="Name of model to test (e.g. model/model.onnx)", default=model)
    parser.add_argument('--output', '-o', help="Location to write outputs (default 'onnx_outputs')", default=output)
    parser.add_argument('--show', '-s', help="Show each inference image", action="store_true")
    args = parser.parse_args()

    dataset = args.input
    if not dataset:
        dataset = os.getenv("INPUT_DATASET")
        if not dataset:
            print("please provide --input or set your INPUT_DATASET environment vairable")
            sys.exit(1)

    test_onnx(dataset, args.model, args.output, show=args.show)
