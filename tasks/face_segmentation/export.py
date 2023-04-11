# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pathlib import Path
import torch
import os
from argparse import ArgumentParser
from archai.discrete_search.search_spaces.config import ArchConfig
from search_space.hgnet import StackedHourglass


def export(checkpoint, model, onnx_file):
    state_dict = checkpoint['state_dict']
    # strip 'model.' prefix off the keys!
    state_dict = dict({(k[6:], state_dict[k]) for k in state_dict})
    model.load_state_dict(state_dict)
    input_shapes = [(1, 3, 256, 256)]
    rand_range = (0.0, 1.0)
    export_kwargs = {'opset_version': 11}
    rand_min, rand_max = rand_range
    sample_inputs = tuple(
        [
            ((rand_max - rand_min) * torch.rand(*input_shape) + rand_min).type("torch.FloatTensor")
            for input_shape in input_shapes
        ]
    )

    torch.onnx.export(
        model,
        sample_inputs,
        onnx_file,
        input_names=[f"input_{i}" for i in range(len(sample_inputs))],
        **export_kwargs,
    )

    print(f'Exported {onnx_file}')


def main():
    parser = ArgumentParser(
        "Converts the final_model.ckpt to final_model.onnx, writing the onnx model to the same folder."
    )
    parser.add_argument('arch', type=Path, help="Path to config.json file describing the model architecture")
    parser.add_argument('--checkpoint', help="Path of the checkpoint to export")

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)

    # get the directory name from args.checkpoint
    output_path = os.path.dirname(os.path.realpath(args.checkpoint))
    base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    onnx_file = os.path.join(output_path, f'{base_name}.onnx')

    arch_config = ArchConfig.from_file(args.arch)
    model = StackedHourglass(arch_config, num_classes=18)
    export(checkpoint, model, onnx_file)


if __name__ == '__main__':
    main()
