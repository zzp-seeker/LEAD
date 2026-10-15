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
import concurrent.futures
import os
import re
import socket
import subprocess
import threading
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

import torch
import zmq
from pydantic import BaseModel, PlainSerializer, PlainValidator, WithJsonSchema
from ray.util.collective import collective

from verl.utils.device import (
    get_device_name,
    get_torch_device,
)

if TYPE_CHECKING:
    from typing import TypeVar

    from typing_extensions import TypedDict

    class FileMeta(TypedDict):
        key: str
        dtype: torch.dtype
        shape: torch.Size
        type: type
        tp_concat_dim: int

    T = TypeVar("T")


def _dt_validate(value: Any) -> torch.dtype:
    if isinstance(value, str):
        if not value.startswith("torch."):
            raise ValueError(f"dtype {value} should start with torch.")
        try:
            value = getattr(torch, value.split(".")[1])
        except AttributeError as e:
            raise ValueError(f"unknown dtype: {value}") from e
    if not isinstance(value, torch.dtype):
        raise TypeError(f"dtype {value} should be torch.dtype, got {type(value)}")
    return value


_TorchDtype = Annotated[
    torch.dtype,
    PlainValidator(_dt_validate),
    PlainSerializer(lambda x: str(x), return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


def _size_validate(value: Any) -> torch.Size:
    if isinstance(value, list | tuple):
        return torch.Size(value)
    if not isinstance(value, torch.Size):
        raise TypeError(f"size {value} should be torch.Size, got {type(value)}")
    return value


_TorchSize = Annotated[
    torch.Size,
    PlainValidator(_size_validate),
    PlainSerializer(lambda x: tuple(x), return_type=tuple),
    WithJsonSchema({"type": "array", "items": {"type": "integer"}}, mode="serialization"),
]


def _tensor_validate(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"tensor {value} should be torch.Tensor, got {type(value)}")


_TorchTensor = Annotated[
    torch.Tensor,
    PlainValidator(_tensor_validate),
]


class ParameterMeta(BaseModel):

    name: str
    dtype: _TorchDtype
    shape: _TorchSize


class MemoryBuffer(BaseModel):

    buffer: _TorchTensor
    size: int
    metas: list[ParameterMeta]


class MemoryBufferMeta(BaseModel):

    size: int
    metas: list[ParameterMeta]


_ALIGN_SIZE = 256


def _align_size(dtype: torch.dtype, shape: torch.Size) -> int:
    return (dtype.itemsize * shape.numel() + _ALIGN_SIZE - 1) // _ALIGN_SIZE * _ALIGN_SIZE


@lru_cache(maxsize=1)
def get_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        print(f"fail to get ip from network interface, fallback to get ip from hostname: {e}")
        return socket.gethostbyname(socket.gethostname())


def npu_generate_uuid() -> str:
    str_pid = str(os.getpid())
    npu_num = 8
    try:
        for npu_id in range(npu_num):
            cmd = ["npu-smi", "info", "-t", "proc-mem", "-i", str(npu_id)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            str_result = str(result.stdout)
            if str_pid in str_result:
                match_chip_count = re.search(r"Chip Count[^\d]*(\d+)", str_result)
                chip_count = int(match_chip_count.group(1))
                search_after_pid = str_result[str_result.find(str_pid) + len(str_pid) :]
                match_chip_id = re.search(r"Chip ID[^\d]*(\d+)", search_after_pid)
                chip_id = int(match_chip_id.group(1))
                return f"{get_ip()}-{npu_id * chip_count + chip_id}"
        raise ValueError("The current process is not running on the npu device")
    except subprocess.CalledProcessError as e:
        raise ValueError("The current process is not running on the npu device") from e


def _get_physical_device_id(device_index: int | None = None) -> str:
    try:
        if get_device_name() == "npu":
            return f"NPU-{npu_generate_uuid()}"
        else:
            return f"GPU-{get_torch_device().get_device_properties(device_index).uuid!s}"
    except AssertionError as e:
        raise ValueError(f"fail to get physical gpu id {device_index}") from e


class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


def _to_flattened_tensor_meta(metas: list[ParameterMeta], offset: int = 0) -> list[FlattenedTensorMetadata]:
    ret = []
    for meta in metas:
        size = _align_size(meta.dtype, meta.shape)
        ret.append(
            {
                "name": meta.name,
                "dtype": meta.dtype,
                "shape": meta.shape,
                "offset": offset,
            }
        )
        offset += size
    return ret


def _extract_weights(
    flatten_metas: list[FlattenedTensorMetadata], buffer: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:

    assert buffer is not None
    weights: list[tuple[str, torch.Tensor]] = []
    for item in flatten_metas:
        shape = item["shape"]
        if isinstance(shape, list | tuple):
            shape = torch.Size(shape)
        assert isinstance(shape, torch.Size)
        dtype, offset = item["dtype"], item["offset"]
        size = dtype.itemsize * shape.numel()
        tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
        weights.append((item["name"], tensor))
    return weights


class CheckpointEngine:

    def __init__(
        self, current_rank: int, actor_ranks: list[int], rollout_ranks: list[int], device_buffer_size_M: int
    ) -> None:
        self.current_rank = current_rank
        self.actor_ranks = actor_ranks
        self.rollout_ranks = rollout_ranks
        self.global_buckets: dict[int, list[MemoryBufferMeta]] = None
        self.device_buffer_size_M = device_buffer_size_M

        self._zmq_ctx = zmq.Context()
        self._zmq_addr_counter: int = 0
        device_index = self.current_rank % get_torch_device().device_count()
        self._device_uuid = _get_physical_device_id(device_index)

    def register_checkpoint(
        self, weights_info: list[tuple[str, torch.Size, torch.dtype]], cpu_named_params: dict[str, torch.Tensor]
    ):
        bucket_size = max(
            self.device_buffer_size_M << 20, max(_align_size(dtype, shape) for _, shape, dtype in weights_info)
        )
        print(
            f"set checkpoint_engine device buffer size: {self.device_buffer_size_M}M, "
            f"and finally set it to {bucket_size >> 20}M considering the largest parameter tensor size"
        )
        self.bucket_size = bucket_size

        if self.global_buckets is None:
            self.global_buckets = {rank: [MemoryBufferMeta(size=0, metas=[])] for rank in self.actor_ranks}

            actor_ranks_size = len(self.actor_ranks)
            assert actor_ranks_size > 0, f"actor_ranks:{self.actor_ranks} should not be empty"
            for param_idx, (param_name, param_shape, param_dtype) in enumerate(weights_info):
                assgin_rank = self.actor_ranks[param_idx % actor_ranks_size]
                param_size = _align_size(param_dtype, param_shape)

                if self.global_buckets[assgin_rank][-1].size + param_size > bucket_size:
                    assert self.global_buckets[assgin_rank][-1].size, (
                        f"global_buckets[{assgin_rank}][-1].size:{self.global_buckets[assgin_rank][-1].size}"
                        " should not be 0"
                    )
                    self.global_buckets[assgin_rank].append(MemoryBufferMeta(size=0, metas=[]))
                self.global_buckets[assgin_rank][-1].metas.append(
                    ParameterMeta(name=param_name, dtype=param_dtype, shape=param_shape)
                )
                self.global_buckets[assgin_rank][-1].size += param_size

        def register_pin_memory(idx: int, size: int) -> tuple[int, torch.Tensor]:
            buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            return idx, buffer

        def register_tensor(buffer: torch.Tensor, offset: int, tensor: torch.Tensor):
            buffer[offset : offset + tensor.nbytes] = tensor.view(-1).view(dtype=torch.uint8)

        memory_buffers = []
        if self.current_rank in self.actor_ranks:
            local_buckets = self.global_buckets[self.current_rank]
            memory_buffers = [
                MemoryBuffer(buffer=torch.empty(0), size=bucket.size, metas=bucket.metas) for bucket in local_buckets
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = [
                    executor.submit(register_pin_memory, idx, bucket.size) for idx, bucket in enumerate(local_buckets)
                ]
                new_futures = []
                for future in concurrent.futures.as_completed(futures):
                    idx, buffer = future.result()
                    assert buffer.numel() == local_buckets[idx].size, (
                        f"buffer numel {buffer.numel()} should be equal to bucket size {local_buckets[idx].size}"
                    )
                    memory_buffers[idx].buffer = buffer
                    print(
                        f"[rank{self.current_rank}] register pin_memory for "
                        f" bucket {idx + 1}/{len(local_buckets)} finished, "
                        f"size {buffer.numel() / 1024 / 1024:.2f}MiB, start to copy tensors to buffer"
                    )
                    offset = 0
                    for meta in local_buckets[idx].metas:
                        name = meta.name
                        tensor = cpu_named_params[name]
                        size = _align_size(tensor.dtype, tensor.shape)
                        assert size == _align_size(meta.dtype, meta.shape), (
                            f"tensor {name} size {size} should be equal to "
                            f"meta size {_align_size(meta.dtype, meta.shape)}"
                        )
                        new_futures.append(executor.submit(register_tensor, buffer, offset, tensor))
                        offset += size
                for future in concurrent.futures.as_completed(new_futures):
                    future.result()

        self.memory_buffers = memory_buffers

    def get_max_buckets_num_per_rank(self):
        assert self.global_buckets is not None
        return max(len(buckets) for buckets in self.global_buckets.values())

    def _bind_zmq_socket(self) -> tuple[zmq.Socket, list[tuple[str, str]]]:

        def zmq_handle(device_uuid: str) -> str:
            return f"ipc://@checkpoint-engine-{device_uuid}-{self._zmq_addr_counter}.sock"

        socket_path = zmq_handle(self._device_uuid)
        socket = self._zmq_ctx.socket(zmq.REQ)
        socket.bind(socket_path)
        self._zmq_addr_counter += 1
        return socket, socket_path

    def update_checkpoint(self, inference_model, group_name: str, overlap_broadcast_and_consume: bool = False):
        try:
            h2d_buffer: torch.Tensor | None = (
                None
                if self.current_rank in self.rollout_ranks
                else torch.empty(self.bucket_size, dtype=torch.uint8, device=get_torch_device().current_device())
            )
            broadcast_load_buffer = torch.empty(
                self.bucket_size * (2 if overlap_broadcast_and_consume else 1),
                dtype=torch.uint8,
                device=get_torch_device().current_device(),
            )
        except Exception:
            print(
                "allocate buffer for update_checkpoint failed, "
                "you may need to reduce "
                "config.async_training.checkpoint_engine.device_buffer_size_M"
            )
            raise

        max_h2d_iter = self.get_max_buckets_num_per_rank()

        if overlap_broadcast_and_consume:
            socket, socket_path = self._bind_zmq_socket()

            def update_weights_from_ipc_(socket_path):
                zmq_ctx = zmq.Context()
                socket = zmq_ctx.socket(zmq.REP)
                socket.connect(socket_path)
                socket.recv_pyobj()
                socket.send(b"")

                while True:
                    payload: tuple[Callable, tuple] | list[FlattenedTensorMetadata] | None = socket.recv_pyobj()
                    if payload is None:
                        get_torch_device().synchronize()
                        socket.send(b"")
                        break
                    assert isinstance(payload, list)
                    if inference_model is not None:
                        inference_model.load_weights(_extract_weights(payload, broadcast_load_buffer))
                    get_torch_device().synchronize()
                    socket.send(b"")

            req_thread = threading.Thread(
                target=update_weights_from_ipc_,
                args=(socket_path,),
            )
            req_thread.start()
            socket.send_pyobj(b"")
            get_torch_device().synchronize()

        gidx = 0
        local_buckets = self.global_buckets.get(self.current_rank, [])

        for i in range(max_h2d_iter):
            if i < len(self.memory_buffers):
                h2d_buffer[: local_buckets[i].size].data.copy_(self.memory_buffers[i].buffer)

            for broadcast_rank, _buckets in self.global_buckets.items():
                if i >= len(_buckets):
                    continue
                bucket = _buckets[i]

                start = gidx % 2 * self.bucket_size if overlap_broadcast_and_consume else 0
                buffer_b: torch.Tensor = broadcast_load_buffer[start : start + bucket.size]
                if broadcast_rank == self.current_rank:
                    buffer_b.data.copy_(h2d_buffer[: bucket.size])

                collective.broadcast(buffer_b, src_rank=broadcast_rank, group_name=group_name)

                if overlap_broadcast_and_consume:
                    socket.recv()
                    collective.barrier(group_name=group_name)
                    socket.send_pyobj(_to_flattened_tensor_meta(bucket.metas, start))
                elif inference_model is not None:
                    named_tensor = _to_flattened_tensor_meta(bucket.metas, 0)
                    inference_model.load_weights(_extract_weights(named_tensor, buffer_b))

                gidx += 1

        if overlap_broadcast_and_consume:
            socket.recv()
            socket.send_pyobj(None)
            socket.recv()
            req_thread.join()
            socket.close()

        collective.barrier(group_name=group_name)
        self.memory_buffers = []
