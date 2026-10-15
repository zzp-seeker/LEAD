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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional

from omegaconf import DictConfig
from pydantic import BaseModel
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayResourcePool, ResourcePoolManager
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig

logger = logging.getLogger(__file__)


class TokenOutput(BaseModel):
    token_ids: list[int]
    """response token ids"""
    log_probs: Optional[list[float]] = None
    """logprobs of response token ids"""
    routed_experts: Optional[Any] = None
    """routed experts of response token ids"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'aborted', or None for unknown"""


class RolloutMode(Enum):
    HYBRID = "hybrid"

    COLOCATED = "colocated"

    STANDALONE = "standalone"


class RolloutReplica(ABC):

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ) -> None:
        self.replica_rank = replica_rank
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = model_config

        self.world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.gpus_per_node = gpus_per_node
        self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0, (
            f"world_size {self.world_size} must be divisible by gpus_per_node {self.gpus_per_replica_node}"
        )
        self.nnodes = self.world_size // self.gpus_per_replica_node
        self.is_reward_model = is_reward_model

        self.rollout_mode: RolloutMode = None
        self.workers: list[ActorHandle] = []
        self.resource_pool: RayResourcePool = None

        self.servers: list[ActorHandle] = []
        self._server_address: str = None
        self._server_handle: ActorHandle = None

    async def init_hybrid(self, worker_group: RayWorkerGroup):
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        await self.launch_servers()

    async def init_colocated(self, resource_pool: RayResourcePool):
        self.rollout_mode = RolloutMode.COLOCATED
        self.resource_pool = resource_pool

        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"rollout_colocate_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_reward_colocate_{self.replica_rank}",
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    async def init_standalone(self):
        self.rollout_mode = RolloutMode.STANDALONE
        resource_pool_name = (
            f"rollout_pool_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_pool_reward_{self.replica_rank}"
        )
        resource_pool_spec = {
            resource_pool_name: [self.gpus_per_replica_node] * self.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=None)
        resource_pool_manager.create_resource_pool()
        self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]


        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=f"rollout_standalone_{self.replica_rank}"
            if not self.is_reward_model
            else f"rollout_reward_standalone_{self.replica_rank}",
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    @abstractmethod
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        raise NotImplementedError

    @abstractmethod
    async def launch_servers(self):
        raise NotImplementedError

    @property
    def server_address(self) -> str:
        return self._server_address

    @property
    def server_handle(self) -> ActorHandle:
        return self._server_handle

    async def wake_up(self):
        await asyncio.gather(*[server.wake_up.remote() for server in self.servers])

    async def sleep(self):
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def clear_kv_cache(self):
        await asyncio.gather(*[server.clear_kv_cache.remote() for server in self.servers])


class RolloutReplicaRegistry:

    _registry: dict[str, Callable[[], type[RolloutReplica]]] = {}

    @classmethod
    def register(cls, name: str, loader: Callable[[], type[RolloutReplica]]) -> None:
        cls._registry[name] = loader

    @classmethod
    def get(cls, name: str) -> type[RolloutReplica]:
        if name not in cls._registry:
            raise ValueError(f"Unknown rollout mode: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]()


def _load_vllm():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    return vLLMReplica


def _load_sglang():
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

    try:
        import vllm
    except ImportError:
        import sys
        import types
        from unittest.mock import Mock

        mock_vllm = types.ModuleType("vllm")

        mock_custom_ops = types.ModuleType("vllm._custom_ops")
        mock_custom_ops.scaled_fp8_quant = Mock()
        mock_vllm._custom_ops = mock_custom_ops

        mock_model_executor = types.ModuleType("vllm.model_executor")
        mock_layers = types.ModuleType("vllm.model_executor.layers")
        mock_activation = types.ModuleType("vllm.model_executor.layers.activation")

        class GeluAndMul:
            pass

        class SiluAndMul:
            pass

        mock_activation.GeluAndMul = GeluAndMul
        mock_activation.SiluAndMul = SiluAndMul
        mock_layers.activation = mock_activation
        mock_model_executor.layers = mock_layers
        mock_vllm.model_executor = mock_model_executor

        sys.modules["vllm"] = mock_vllm
        sys.modules["vllm._custom_ops"] = mock_custom_ops
        sys.modules["vllm.model_executor"] = mock_model_executor
        sys.modules["vllm.model_executor.layers"] = mock_layers
        sys.modules["vllm.model_executor.layers.activation"] = mock_activation

    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica

    del os.environ["SGLANG_USE_CPU_ENGINE"]
    return SGLangReplica


RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)


def get_rollout_replica_class(rollout: str) -> type[RolloutReplica]:
    return RolloutReplicaRegistry.get(rollout)
