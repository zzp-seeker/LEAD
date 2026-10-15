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
from abc import ABC, abstractmethod
from typing import Generator, TypedDict

import torch


class TensorMeta(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


class CheckpointEngineRegistry:

    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):

        def wrapper(cls: type["CheckpointEngine"]):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine {backend} not registered")
        return cls._registry[backend](*args, **kwargs)


class CheckpointEngine(ABC):

    @abstractmethod
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError


class CheckpointEngineWithCache(CheckpointEngine):

    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):

    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        self.weights = weights

    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield from self.weights
        self.weights = None
