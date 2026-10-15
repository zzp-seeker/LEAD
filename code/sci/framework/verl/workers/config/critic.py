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
import warnings
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import BaseModelConfig, CheckpointConfig
from verl.utils.profiler import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = ["CriticConfig", "FSDPCriticConfig", "McoreCriticConfig", "FSDPCriticModelCfg"]


@dataclass
class CriticConfig(BaseConfig):

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_micro_batch_size_per_gpu",
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "model_config",
    }

    strategy: str = MISSING
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    enable: Optional[bool] = None
    rollout_n: int = 1
    ppo_mini_batch_size: int = 1
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 32768
    forward_max_token_len_per_gpu: int = 32768
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_max_token_len_per_gpu: int = 32768
    ppo_epochs: int = 1
    data_loader_seed: int = 1
    shuffle: bool = True
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    ppo_micro_batch_size: Optional[int] = None
    engine: BaseConfig = field(default_factory=BaseConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    model_config: HFModelConfig = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        assert self.strategy != MISSING

        if self.model_config is None:
            warnings.warn("using model in Critic Config is deprecated, please use model_config instead", stacklevel=2)
            self.model_config = HFModelConfig(
                path=self.model.path,
                tokenizer_path=self.model.tokenizer_path,
                override_config=self.model.override_config,
                external_lib=self.model.external_lib,
                trust_remote_code=self.model.trust_remote_code,
            )

        if not self.use_dynamic_bsz:
            self._check_mutually_exclusive(self.ppo_micro_batch_size, self.ppo_micro_batch_size_per_gpu, "critic")

            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"[critic] ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"critic.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        param = "micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreCriticConfig(CriticConfig):

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    load_weight: bool = True

    def validate(self, n_gpus: int, train_batch_size: int):
        super().validate(n_gpus, train_batch_size)


@dataclass
class FSDPCriticConfig(CriticConfig):

    _mutable_fields = CriticConfig._mutable_fields | {
        "forward_micro_batch_size",
        "forward_micro_batch_size_per_gpu",
    }

    strategy: str = "fsdp"
    forward_micro_batch_size: int = 1
    forward_micro_batch_size_per_gpu: int = 1
    ulysses_sequence_parallel_size: int = 1
    grad_clip: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        if self.strategy in {"fsdp", "fsdp2"}:
            if self.ulysses_sequence_parallel_size > 1:
                if not self.model.get("use_remove_padding", False):
                    raise ValueError(
                        "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        super().validate(n_gpus, train_batch_size)

        if not self.use_dynamic_bsz:
            sp_size = self.ulysses_sequence_parallel_size
            if self.ppo_micro_batch_size is not None:
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"critic.ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )


@dataclass
class FSDPCriticModelCfg(BaseModelConfig):

    use_shm: bool = False
    enable_activation_offload: bool = False
    use_remove_padding: bool = False
    enable_gradient_checkpointing: bool = True
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str | list[str] = "all-linear"
    tiled_mlp: dict = field(default_factory=lambda: {"enabled": False, "num_shards": 4})
