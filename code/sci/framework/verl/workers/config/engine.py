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
from typing import Any, Literal, Optional

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig

from ...utils.profiler import ProfilerConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = ["FSDPEngineConfig", "McoreEngineConfig", "TrainingWorkerConfig", "VeOmniEngineConfig"]


@dataclass
class EngineConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "use_dynamic_bsz",
        "max_token_len_per_gpu",
        "micro_batch_size_per_gpu",
        "infer_max_token_len_per_gpu",
        "infer_micro_batch_size_per_gpu",
        "use_fused_kernels",
        "use_remove_padding",
    }

    param_offload: bool = False
    optimizer_offload: bool = False
    grad_offload: bool = False
    forward_only: bool = False
    strategy: str = None
    dtype: str = "bfloat16"
    use_dynamic_bsz: bool = True
    max_token_len_per_gpu: int = None
    micro_batch_size_per_gpu: int = None
    infer_max_token_len_per_gpu: int = None
    infer_micro_batch_size_per_gpu: int = None
    use_fused_kernels: bool = False
    use_remove_padding: bool = True

    seed: int = 42

    full_determinism: bool = False

    def __post_init__(self):
        pass


@dataclass
class McoreEngineConfig(EngineConfig):

    _mutable_fields = EngineConfig._mutable_fields | {"sequence_parallel"}
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: Optional[str] = None
    dist_checkpointing_prefix: str = ""
    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    override_mcore_model_config: dict[str, Any] = field(default_factory=dict)
    use_mbridge: bool = True
    vanilla_mbridge: bool = True
    strategy: str = "megatron"

    def __post_init__(self) -> None:
        super().__post_init__()
        """config validation logics go here"""
        assert self.strategy == "megatron"
        assert self.dtype in ["bfloat16", "float16"], f"dtype {self.dtype} not supported"
        if self.tensor_model_parallel_size == 1:
            warnings.warn("set sequence parallel to false as TP size is 1", stacklevel=2)
            self.sequence_parallel = False


@dataclass
class FSDPEngineConfig(EngineConfig):

    _mutable_fields = EngineConfig._mutable_fields | {"ulysses_sequence_parallel_size"}

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False
    mixed_precision: Optional[dict[str, Any]] = None
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "fsdp"

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["fsdp", "fsdp2"], f"strategy {self.strategy} not supported"


@dataclass
class VeOmniEngineConfig(EngineConfig):

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    forward_prefetch: bool = False
    use_orig_params: bool = False
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "veomni"
    data_parallel_size: int = 1
    data_parallel_replicate_size: int = 1
    data_parallel_shard_size: int = 1
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    ulysses_parallel_size: int = 1
    data_parallel_mode: Literal["ddp", "fsdp1", "fsdp2"] = "fsdp"
    seed: int = 42
    full_determinism: bool = False
    mixed_precision: bool = False
    init_device: str = "meta"
    enable_full_shard: bool = False
    ckpt_manager: Literal["dcp"] = "dcp"
    load_checkpoint_path: Optional[str] = None
    enable_fsdp_offload: bool = False
    enable_reentrant: bool = False
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2", "native-sparse"]] = "flash_attention_2"
    moe_implementation: Optional[Literal[None, "eager", "fused"]] = "eager"
    force_use_huggingface: bool = False
    activation_gpu_limit: float = 0.0
    basic_modules: Optional[list[str]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["veomni"], f"strategy {self.strategy} not supported"


@dataclass
class TrainingWorkerConfig(BaseConfig):
    model_type: str = None
    model_config: HFModelConfig = None
    engine_config: EngineConfig = None
    optimizer_config: OptimizerConfig = None
    checkpoint_config: CheckpointConfig = None
    profiler_config: ProfilerConfig = None
