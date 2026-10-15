# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import cupy as cp
import ray
import ray.util.collective as collective
import torch
import zmq

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int


class BroadcastOperation:

    def __init__(
        self,
        rank: int,
        group_name: str,
        bucket: cp.ndarray | torch.Tensor,
        metadata: dict[str, TensorMeta],
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        self.rank = rank
        self.group_name = group_name
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()

        collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("nccl")
class NCCLCheckpointEngine(CheckpointEngine):

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype

        self.is_master = is_master
        self.topic = "bucket_metadata"
        if self.is_master:
            self._start_zmq_server()

    def prepare(self) -> MasterMetadata:
        if self.is_master:
            self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
        else:
            self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
            self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")

        return MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port) if self.is_master else None

    def finish(self):
        if self.rebuild_group:
            if self.rank >= 0:
                collective.destroy_collective_group(self.group_name)
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None

    def _start_zmq_server(self):
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port, self.listen_sock = get_free_port(self.ip)

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.listen_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.listen_port}"

        self.socket.bind(address)

    def _connect_zmq_client(self, metadata: MasterMetadata):
        assert not self.is_master, "Master process should not connect to other processes."
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return

        if self.rebuild_group or not collective.is_group_initialized(self.group_name):
            collective.init_collective_group(world_size, rank, "nccl", self.group_name)
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self.world_size}"
            )

        if self.rank > 0:
            self._connect_zmq_client(master_metadata)
        collective.barrier(self.group_name)

        logger.info(f"init_process_group rank: {self.rank}, world_size: {self.world_size}")

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."

        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        for name, weight in weights:
            weight = weight.to(self.rollout_dtype)

            if offset + weight.nbytes > self.bucket_size:
                torch.cuda.synchronize()

                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    group_name=self.group_name,
                    bucket=send_buf,
                    metadata={"bucket_meta": bucket_meta, "is_last": False},
                    socket=self.socket,
                    topic=self.topic,
                )

                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            send_buf[offset : offset + weight.nbytes] = cp.asarray(weight.view(-1).view(torch.uint8))
            offset += weight.nbytes

        torch.cuda.synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            group_name=self.group_name,
            bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True},
            socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()
        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        assert self.rank > 0, "Rank 0 should not receive weights."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            group_name=self.group_name,
            bucket=recv_buf,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        send_buf, recv_buf = recv_buf, send_buf
        while not metadata["is_last"]:
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                group_name=self.group_name,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
            )

            for name, meta in metadata["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = send_buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata["bucket_meta"])

            torch.cuda.synchronize()
            send_buf, recv_buf = recv_buf, send_buf

        for name, meta in metadata["bucket_meta"].items():
            dtype, shape = meta["dtype"], meta["shape"]
            size = dtype.itemsize * shape.numel()
            tensor = send_buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
            yield name, tensor

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
