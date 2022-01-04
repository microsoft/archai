# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Measures GPU-related information.
"""

import collections
import math
import os
import pathlib
import re
from typing import List, Optional

import pynvml

pynvml.nvmlInit()


def system_get_driver_version() -> str:
    """Gathers the GPU driver version.

    Returns:
        (str): Driver version of current GPU.

    """

    return pynvml.nvmlSystemGetDriverVersion()


def device_get_count() -> int:
    """Gathers the amount of GPU-based devices.

    Returns:
        (int): Number of GPU devices.

    """

    return pynvml.nvmlDeviceGetCount()


class Device:
    """Wraps GPU affinity information.

    """

    # assume nvml returns list of 64 bit ints
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)

    def __init__(self, device_idx: int) -> None:
        """Overrides initialization method.

        Args:
            device_idx: Device identifier.

        """

        super().__init__()

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self) -> str:
        """Gathers the name of current GPU.

        Returns:
            (str): Name of current GPU.

        """

        return pynvml.nvmlDeviceGetName(self.handle)

    def get_cpu_affinity(self) -> List[int]:
        """Gets the CPU affinity.

        Returns:
            (List[int]): CPU affinity.

        """
        
        affinity_string = ''

        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, Device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = '{:064b}'.format(j) + affinity_string

        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        ret = [i for i, e in enumerate(affinity_list) if e != 0]

        return ret


def set_socket_affinity(gpu_id: int) -> None:
    """Sets the socket affinity on the supplied GPU.

    Args:
        gpu_id: GPU identifier.

    """

    dev = Device(gpu_id)

    affinity = dev.get_cpu_affinity()

    os.sched_setaffinity(0, affinity)


def set_single_affinity(gpu_id: int) -> None:
    """Sets a single affinity on the supplied GPU.

    Args:
        gpu_id: GPU identifier.

    """

    dev = Device(gpu_id)

    affinity = dev.get_cpu_affinity()

    os.sched_setaffinity(0, affinity[:1])


def set_single_unique_affinity(gpu_id: int, nproc_per_node: int) -> None:
    """Sets a single and unique affinity on supplied GPU.

    Args:
        gpu_id: GPU identifier.
        nproc_per_node: Number of processes per node.

    """

    devices = [Device(i) for i in range(nproc_per_node)]
    socket_affinities = [dev.get_cpu_affinity() for dev in devices]

    siblings_list = get_thread_siblings_list()
    siblings_dict = dict(siblings_list)

    # remove siblings
    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities[idx] = list(set(socket_affinity) - set(siblings_dict.values()))

    affinities = []
    assigned = []

    for socket_affinity in socket_affinities:
        for core in socket_affinity:
            if core not in assigned:
                affinities.append([core])
                assigned.append(core)
                break

    os.sched_setaffinity(0, affinities[gpu_id])


def set_socket_unique_affinity(gpu_id: int, nproc_per_node: int, mode: str) -> None:
    """Sets a socket and unique affinity on supplied GPU.

    Args:
        gpu_id: GPU identifier.
        nproc_per_node: Number of processes per node.
        mode: Affinity mode.

    """

    device_ids = [Device(i) for i in range(nproc_per_node)]
    socket_affinities = [dev.getCpuAffinity() for dev in device_ids]

    siblings_list = get_thread_siblings_list()
    siblings_dict = dict(siblings_list)

    # remove siblings
    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities[idx] = list(set(socket_affinity) - set(siblings_dict.values()))

    socket_affinities_to_device_ids = collections.defaultdict(list)

    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities_to_device_ids[tuple(socket_affinity)].append(idx)

    for socket_affinity, device_ids in socket_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        cores_per_device = len(socket_affinity) // devices_per_group

        for group_id, device_id in enumerate(device_ids):
            if device_id == gpu_id:
                if mode == 'interleaved':
                    affinity = list(socket_affinity[group_id::devices_per_group])
                elif mode == 'continuous':
                    affinity = list(socket_affinity[group_id*cores_per_device:(group_id+1)*cores_per_device])
                else:
                    raise RuntimeError('Unknown set_socket_unique_affinity mode')

                # reintroduce siblings
                affinity += [siblings_dict[aff] for aff in affinity if aff in siblings_dict]
                os.sched_setaffinity(0, affinity)


def get_thread_siblings_list() -> List[str]:
    """Gets a list of current thread siblings.

    Returns:
        (List[str]): Thread siblings.

    """

    path = '/sys/devices/system/cpu/cpu*/topology/thread_siblings_list'
    thread_siblings_list = []
    pattern = re.compile(r'(\d+)\D(\d+)')

    for fname in pathlib.Path(path[0]).glob(path[1:]):
        with open(fname) as f:
            content = f.read().strip()
            res = pattern.findall(content)

            if res:
                pair = tuple(map(int, res[0]))
                thread_siblings_list.append(pair)

    return thread_siblings_list


def set_affinity(gpu_id: int, nproc_per_node: int, mode: Optional[str] = 'socket') -> None:
    """Sets affinity on supplied GPU.

    Args:
        gpu_id: GPU identifier.
        nproc_per_node: Number of processes per node.
        mode: Affinity mode.

    """

    if mode == 'socket':
        set_socket_affinity(gpu_id)
    elif mode == 'single':
        set_single_affinity(gpu_id)
    elif mode == 'single_unique':
        set_single_unique_affinity(gpu_id, nproc_per_node)
    elif mode == 'socket_unique_interleaved':
        set_socket_unique_affinity(gpu_id, nproc_per_node, 'interleaved')
    elif mode == 'socket_unique_continuous':
        set_socket_unique_affinity(gpu_id, nproc_per_node, 'continuous')
    else:
        raise RuntimeError('Unknown affinity mode')

    affinity = os.sched_getaffinity(0)
    
    return affinity
