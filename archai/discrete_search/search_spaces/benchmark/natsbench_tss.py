# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import re
import warnings
from pathlib import Path
from typing import Any, List, Optional

import nats_bench
import numpy as np
import torch
import yaml
from overrides import overrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_space import (
    BayesOptSearchSpace,
    EvolutionarySearchSpace,
)


class NatsbenchTssSearchSpace(EvolutionarySearchSpace, BayesOptSearchSpace):
    """Search space for NATS-Bench-TSS."""

    OPS = ["none", "avg_pool_3x3", "nor_conv_1x1", "nor_conv_3x3", "skip_connect"]

    def __init__(self, natsbench_location: str, base_dataset: str, seed: Optional[int] = 1) -> None:
        """Initialize the search space.

        Args:
            natsbench_location: Path to the NATS-Bench-TSS dataset.
            base_dataset: Base dataset used for training the models.
            seed: Random seed.

        """

        try:
            from xautodl.models import get_cell_based_tiny_net
        except ImportError:
            raise ImportError(
                "xautodl installation not found. "
                "Please install it using `pip install git+https://github.com/D-X-Y/AutoDL-Projects/`"
            )

        self.natsbench_location = Path(natsbench_location)
        self.base_dataset = base_dataset
        assert base_dataset in [
            "cifar10",
            "cifar100",
            "ImageNet16-120",
        ], "`base_dataset` must be one of ['cifar10', 'cifar100', 'ImageNet16-120']"

        if not self.natsbench_location.exists():
            raise FileNotFoundError(
                "The provided path to `natsbench_location` (" f"{self.natsbench_location.absolute()}) does not exist"
            )

        self.api = nats_bench.create(natsbench_location, "tss", fast_mode=True, verbose=False)
        self.rng = random.Random(seed)
        self.archid_pattern = re.compile("natsbench-tss-([0-9]+)")

    def _get_op_list(self, string: str) -> List[str]:
        """Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py."""

        # Given a string, get the list of operations
        tokens = string.split("|")
        ops = [t.split("~")[0] for i, t in enumerate(tokens) if i not in [0, 2, 5, 9]]

        return ops

    def _get_string_from_ops(self, ops: List[str]) -> str:
        """Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py."""

        # Given a list of operations, get the string
        strings = ["|"]
        nodes = [0, 0, 1, 0, 1, 2]

        for i, op in enumerate(ops):
            strings.append(op + "~{}|".format(nodes[i]))
            if i < len(nodes) - 1 and nodes[i + 1] == 0:
                strings.append("+|")
        return "".join(strings)

    def model_from_natsbench_tss(self, natsbench_id: int) -> Any:
        """Get a model from NATS-Bench-TSS dataset.

        Args:
            natsbench_id: NATS-Bench-TSS identifier.

        Returns:
            Model from NATS-Bench-TSS dataset.

        """

        config = self.api.get_net_config(natsbench_id, self.base_dataset)
        return get_cell_based_tiny_net(config)

    @overrides
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        yaml.safe_dump({"archid": model.archid, **model.metadata}, open(path, "w", encoding="utf-8"))

    @overrides
    def load_arch(self, path: str) -> ArchaiModel:
        metadata = yaml.safe_load(open(path, encoding="utf-8"))
        natsbenchid = self.archid_pattern.match(metadata["archid"])

        if not natsbenchid:
            raise ValueError(f'Architecture {metadata["archid"]} does not belong to `NatsbenchTssSearchSpace`. ')

        if metadata["dataset"] != self.base_dataset:
            warnings.warn(
                f'Architecture loaded from {path} was saved using a different dataset ({metadata["dataset"]})'
                f" than `NatsbenchTssSearchSpace` base dataset ({self.base_dataset})"
            )

        idx = int(natsbenchid.group(1))

        return ArchaiModel(
            arch=self.model_from_natsbench_tss(idx),
            archid=f"natsbench-tss-{idx}",
            metadata={"dataset": self.base_dataset},
        )

    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        model.arch.load_state_dict(torch.load(path))

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        torch.save(model.arch.state_dict(), path)

    @overrides
    def random_sample(self) -> ArchaiModel:
        idx = self.rng.randint(0, len(self.api))

        return ArchaiModel(
            arch=self.model_from_natsbench_tss(idx),
            archid=f"natsbench-tss-{idx}",
            metadata={"dataset": self.base_dataset},
        )

    @overrides
    def mutate(self, model: ArchaiModel) -> ArchaiModel:
        """Reused from https://github.com/naszilla/naszilla/blob/master/naszilla/nas_bench_201/cell_201.py."""

        # First get the string representation of the current architecture
        natsbenchid = self.archid_pattern.match(model.archid)

        if not natsbenchid:
            raise ValueError(f"Architecture {model.archid} does not belong to the `NatsbenchTssSearchSpace`. ")

        natsbenchid = int(natsbenchid.group(1))
        string_rep = self.api.get_net_config(natsbenchid, self.base_dataset)["arch_str"]

        nbhd_strs = []
        ops = self._get_op_list(string_rep)

        for i in range(len(ops)):
            available = [op for op in self.OPS if op != ops[i]]

            for op in available:
                new_ops = ops.copy()
                new_ops[i] = op
                new_arch_str = self._get_string_from_ops(new_ops)
                nbhd_strs.append(new_arch_str)

        # Picks one neighbor architecture as the mutation
        mutation_str = random.choice(nbhd_strs)
        mutation_natsbenchid = self.api.archstr2index[mutation_str]

        return ArchaiModel(
            arch=self.model_from_natsbench_tss(mutation_natsbenchid),
            archid=f"natsbench-tss-{mutation_natsbenchid}",
            metadata={"dataset": self.base_dataset},
        )

    @overrides
    def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
        raise NotImplementedError

    @overrides
    def encode(self, arch: ArchaiModel) -> np.ndarray:
        enc_dict = {
            "none": [0, 0, 0, 0],
            "avg_pool_3x3": [1, 0, 0, 0],
            "nor_conv_1x1": [0, 1, 0, 0],
            "nor_conv_3x3": [0, 0, 1, 0],
            "skip_connect": [0, 0, 0, 1],
        }

        # Gets string repr for `arch`
        natsbenchid = self.archid_pattern.match(arch.archid)
        if not natsbenchid:
            raise ValueError(f"Architecture {arch.archid} does not belong" " to `NatsbenchTssSearchSpace`. ")

        arch_str = self.api[int(natsbenchid.group(1))]
        arch_ops = re.findall(r"([^\|\~\+]+)~\d", arch_str)

        return np.hstack([np.array(enc_dict[op_name]) for op_name in arch_ops])
