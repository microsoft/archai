# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
import math
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union

import torch
import yaml
from torch import nn

from archai.discrete_search.search_spaces.cv.segmentation_dag.ops import OPS, Block


class SegmentationDagModel(torch.nn.Module):
    """Model defined by a directed acyclic graph (DAG) of operations."""

    def __init__(
        self,
        graph: List[Dict[str, Any]],
        channels_per_scale: Dict[str, Any],
        post_upsample_layers: Optional[int] = 1,
        stem_stride: Optional[int] = 2,
        img_size: Optional[Tuple[int, int]] = (256, 256),
        nb_classes: Optional[int] = 19,
    ) -> None:
        """Initialize the model.

        Args:
            graph: List of dictionaries with the following keys:
                * name: Name of the node.
                * op: Name of the operation used to process the node.
                * inputs: List of input nodes.
                * scale: Scale of the node (higher means smaller resolutions).
            channels_per_scale: Dictionary with the number of channels that should be used
                for each scale value, e.g: {1: 32, 2: 64, 4: 128} or a dictionary containing
                `base_channels`, `delta_channels` and optionally a `mult_delta` flag.
                For instance, {'base_channels': 24, 'delta_channels': 2}, is equivalent to
                {1: 24, 2: 26, 4: 28, 8: 30, 16: 32}, and {'base_channels': 24, 'delta_channels': 2,
                mult_delta: True} is equivalent to {1: 24, 2: 48, 4: 96, 8: 192, 16: 384}.
            post_upsample_layers: Number of post-upsample layers.
            stem_strid: Stride of the first convolution.
            img_size: Image size (width, height).
            nb_classes: Number of classes for segmentation.

        """

        super().__init__()
        assert img_size[0] % 32 == 0 and img_size[1] % 32 == 0, "Image size must be a multiple of 32"

        self.graph = OrderedDict([(n["name"], n) for n in graph])
        self.node_names = [n["name"] for n in self.graph.values()]
        self.channels_per_scale = self._get_channels_per_scale(channels_per_scale)
        self.edge_dict = nn.ModuleDict(self._get_edge_list(self.graph, self.channels_per_scale))
        self.stem_stride = stem_stride
        self.img_size = img_size
        self.nb_classes = nb_classes
        self.post_upsample_layers = post_upsample_layers

        # Checks if the edges are in topological order
        self._validate_edges(self.edge_dict)

        # Stem block
        stem_ch = self.channels_per_scale[self.graph["input"]["scale"]]
        self.stem_block = OPS["conv3x3"](3, stem_ch, stride=self.stem_stride)

        # Upsample layers
        w, h = self.img_size
        self.up = nn.Upsample(size=(h, w), mode="nearest")
        output_ch = self.channels_per_scale[self.graph["output"]["scale"]]

        self.post_upsample = nn.Sequential(
            *[
                OPS["conv3x3"](
                    output_ch if i == 0 else self.channels_per_scale[1], self.channels_per_scale[1], stride=1
                )
                for i in range(self.post_upsample_layers)
            ]
        )

        # Classifier
        self.classifier = nn.Conv2d(stem_ch, self.nb_classes, kernel_size=1)

    @classmethod
    def _get_channels_per_scale(
        cls,
        ch_per_scale: Dict[str, Any],
        max_downsample_factor: Optional[int] = 16,
        remove_spec: Optional[bool] = False,
    ) -> Dict[str, Any]:
        ch_per_scale = deepcopy(ch_per_scale)
        scales = [1, 2, 4, 8, 16]
        scales = [s for s in scales if s <= max_downsample_factor]

        # Builds `ch_per_scale` using `base_channels` and `delta_channels`
        ch_per_scale["mult_delta"] = ch_per_scale.get("mult_delta", False)
        assert "base_channels" in ch_per_scale
        assert "delta_channels" in ch_per_scale

        assert len(ch_per_scale.keys()) == 3, "Must specify only `base_channels`, `delta_channels` and `mult_delta`"

        if ch_per_scale["mult_delta"]:
            ch_per_scale.update(
                {
                    scale: ch_per_scale["base_channels"] * ch_per_scale["delta_channels"] ** i
                    for i, scale in enumerate(scales)
                }
            )
        else:
            ch_per_scale.update(
                {
                    scale: ch_per_scale["base_channels"] + ch_per_scale["delta_channels"] * i
                    for i, scale in enumerate(scales)
                }
            )

        if remove_spec:
            ch_per_scale.pop("base_channels", None)
            ch_per_scale.pop("delta_channels", None)
            ch_per_scale.pop("mult_delta", None)

        return ch_per_scale

    def _get_edge_list(
        self, graph: OrderedDict, channels_per_scale: Dict[str, Any]
    ) -> MutableMapping[Tuple[str, str], nn.Module]:
        assert "input" in graph
        assert "output" in graph

        edges = [
            (in_node, node["name"]) for node in graph.values() if node["name"] != "input" for in_node in node["inputs"]
        ]

        # Returns an `OrderedDict` with the mapping "in_node-out_node": nn.Module
        return OrderedDict(
            [
                (
                    f"{i}-{o}",
                    Block(
                        in_ch=channels_per_scale[graph[i]["scale"]],
                        out_ch=channels_per_scale[graph[o]["scale"]],
                        in_scale=graph[i]["scale"],
                        out_scale=graph[o]["scale"],
                        op_name=graph[i]["op"],
                    ),
                )
                for i, o in edges
            ]
        )

    def _validate_edges(self, edge_dict: MutableMapping[Tuple[str, str], nn.Module]) -> None:
        visited_nodes = {"input"}

        for edge in edge_dict.keys():
            in_node, out_node = edge.split("-")
            visited_nodes.add(out_node)

            assert (
                in_node in visited_nodes
            ), "SegmentationModel received a list of nodes that is not in topological order"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = {node_name: 0 for node_name in self.node_names}
        inputs["input"] = self.stem_block(x)

        for edge, module in self.edge_dict.items():
            in_node, out_node = edge.split("-")
            inputs[out_node] = inputs[out_node] + module(inputs[in_node])

        output = self.post_upsample(self.up(inputs["output"]))
        return self.classifier(output)

    def validate_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Checks if the constructed model is working as expected.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.

        """

        in_nodes = set()
        res_w, res_h = [d // self.stem_stride for d in self.img_size]

        inputs = {node_name: 0 for node_name in self.node_names}
        inputs["input"] = self.stem_block(x)

        for edge, module in self.edge_dict.items():
            in_node, out_node = edge.split("-")
            in_nodes.add(in_node)

            # Checks if the resolution of each node is correct
            assert inputs[in_node].shape[2] == int(
                res_h // self.graph[in_node]["scale"]
            ), "Input resolution does not match the node resolution."
            assert inputs[in_node].shape[3] == int(
                res_w // self.graph[in_node]["scale"]
            ), "Input resolution does not match the node resolution."

            inputs[out_node] = inputs[out_node] + module(inputs[in_node])

            assert (
                inputs[out_node].shape[1] == self.channels_per_scale[self.graph[out_node]["scale"]]
            ), "Output channel does not match the node channel scale."

        assert all(
            node in in_nodes for node in set(self.graph.keys()) - {"output"}
        ), f'Unused nodes were detected: {set(self.graph.keys()) - in_nodes - set(["output"])}.'

        output = self.post_upsample(self.up(inputs["output"]))
        return self.classifier(output)

    @classmethod
    def from_file(
        cls, config_file: Union[str, Path], img_size: Optional[Tuple[int, int]] = 256, nb_classes: Optional[int] = 19
    ) -> SegmentationDagModel:
        """Creates a SegmentationArchaiModel from a YAML config file.

        Args:
            config_file: Path to the YAML config file, following the format:
                >>> post_upsample_layers: 2
                >>> channels_per_scale:
                >>>     1: 32
                >>>     2: 64
                >>> architecture:
                >>>     - name: input
                >>>       scale: 1
                >>>       op: conv3x3
                >>>       inputs: null
                >>>     - name: node0
                >>>       scale: 2
                >>>       op: conv5x5
                >>>       inputs: [input]
                >>>     - name: output
                >>>       scale: 4
                >>>       op: conv3x3
                >>>       inputs: [node0, node1]
            img_size: The size of the input image.
            nb_classes: The number of classes in the dataset.

        Returns:
            A `SegmentationArchaiModel` instance.

        """

        config_file = Path(config_file)
        assert config_file.is_file()

        config_dict = yaml.safe_load(open(config_file))
        return cls(
            config_dict["architecture"],
            config_dict["channels_per_scale"],
            config_dict["post_upsample_layers"],
            img_size=img_size,
            nb_classes=nb_classes,
        )

    def view(self) -> Any:
        """Visualizes the architecture using graphviz.

        Returns:
            A graphviz object.

        """

        import graphviz

        scales = []
        dot = graphviz.Digraph("architecture", graph_attr={"splines": "true", "overlap": "true"})
        dot.engine = "neato"

        for i, node in enumerate(self.node_names):
            scales.append(self.graph[node]["scale"])
            dot.node(node, label=self.graph[node]["op"], pos=f"{i*1.5 + 2},-{math.log2(2*scales[-1])}!")

        for scale in sorted(list(set(scales))):
            dot.node(
                f"scale-{scale}",
                label=f"scale={2*scale}, ch={self.channels_per_scale[scale]}",
                pos=f"-1,-{math.log2(2*scale)}!",
            )

        for edge in self.edge_dict:
            in_node, out_node = edge.split("-")
            dot.edge(in_node, out_node)

        # Adds post upsample
        dot.node("upsample", label=f"Upsample + {self.post_upsample_layers} x Conv 3x3", pos=f"{i*1.5 + 2},0!")
        dot.edge("output", "upsample")

        # Shows the graph
        return dot

    def to_config(self) -> Dict[str, Any]:
        """Converts the model to a configuration dictionary.

        Returns:
            A configuration dictionary.

        """

        ch_map = self.channels_per_scale

        if "base_channels" in ch_map:
            ch_map = {"base_channels": ch_map["base_channels"], "delta_channels": ch_map["delta_channels"]}

            # We only put the `mult_delta` flag in config dict if it's active
            if self.channels_per_scale["mult_delta"]:
                ch_map["mult_delta"] = True

        return {
            "post_upsample_layers": int(self.post_upsample_layers),
            "channels_per_scale": ch_map,
            "architecture": list(self.graph.values()),
        }

    def to_file(self, path: str) -> None:
        """Saves the model to a YAML config file.

        Args:
            path: Path to the YAML config file.

        """

        content = self.to_config()

        with open(path, "w") as fp:
            fp.write(yaml.dump(content))

        m = SegmentationDagModel.from_file(path, self.img_size, self.nb_classes)
        assert content["architecture"] == list(m.graph.values())
        assert content["post_upsample_layers"] == len(self.post_upsample)
        assert all(m.channels_per_scale[k] == v for k, v in content["channels_per_scale"].items())

    def to_hash(self) -> str:
        """Generates a hash for the model.

        Returns:
            A hash string.

        """

        config = self.to_config()
        arch_str = json.dumps(config, sort_keys=True, ensure_ascii=True)

        return sha1(arch_str.encode("ascii")).hexdigest() + f"_{self.img_size[0]}_{self.img_size[1]}"
