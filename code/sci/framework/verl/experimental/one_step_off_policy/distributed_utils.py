# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ipaddress
import socket
from datetime import timedelta

import vllm
from torch.distributed import TCPStore
from vllm.distributed.utils import StatelessProcessGroup

from verl.utils.device import is_npu_available


@staticmethod
def create(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    data_expiration_seconds: int = 3600,
    store_timeout: int = 300,
) -> "StatelessProcessGroup":
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        address_family = socket.AF_INET6
    except (ipaddress.AddressValueError, ValueError):
        address_family = socket.AF_INET

    launch_server = rank == 0
    if launch_server:
        listen_socket = socket.socket(address_family, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if address_family == socket.AF_INET6:
            try:
                listen_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            except (AttributeError, OSError):
                pass

        bind_host = host.strip("[]")
        listen_socket.bind((bind_host, port))
        listen_socket.listen()
        listen_fd = listen_socket.fileno()
    else:
        listen_socket = None
        listen_fd = None

    store = TCPStore(
        host_name=host,
        port=port,
        world_size=world_size,
        is_master=launch_server,
        timeout=timedelta(seconds=store_timeout),
        use_libuv=False,
        master_listen_fd=listen_fd,
    )

    return StatelessProcessGroup(
        rank=rank,
        world_size=world_size,
        store=store,
        socket=listen_socket,
        data_expiration_seconds=data_expiration_seconds,
    )


vllm.distributed.utils.StatelessProcessGroup.create = create


def vllm_stateless_init_process_group(master_address, master_port, rank, world_size, device):
    if is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as PyNcclCommunicator,
        )
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
