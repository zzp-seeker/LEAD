# Copyright 2025 Individual Contributor: Thibaut Barroyer
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
from __future__ import annotations

import inspect
import multiprocessing
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast

import ray
import torch

from verl.utils.reward_score import default_compute_score
from verl.utils.transferqueue_utils import tqbridge

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
    from verl.trainer.config.config import ModuleConfig, RewardManagerConfig
    from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
else:
    try:
        from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
    except ImportError:
        RewardManagerBase = None


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:

    reward_fn_config = config.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None

    fn_name = reward_fn_config.get("name")
    assert fn_name is not None

    from verl.utils.import_utils import load_extern_object

    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:

    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    reward_manager_cfg: RewardManagerConfig = config.reward_manager
    reward_manager_cls: type[AbstractRewardManager]
    if reward_manager_cfg.source == "register":
        from verl.workers.reward_manager import get_reward_manager_cls

        reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)
    elif reward_manager_cfg.source == "importlib":
        from verl.utils.import_utils import load_extern_object

        module_cfg: ModuleConfig | None = reward_manager_cfg.module
        assert module_cfg is not None and module_cfg.path is not None, (
            f"Module path is required when {reward_manager_cfg.source=}, but got {module_cfg=}"
        )
        reward_manager_cls_name = reward_manager_cfg.name
        reward_manager_cls = cast(
            "type[AbstractRewardManager]",
            load_extern_object(module_path=module_cfg.path, object_name=reward_manager_cls_name),
        )

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    if RewardManagerBase is not None and issubclass(reward_manager_cls, RewardManagerBase):
        return reward_manager_cls(
            config=config,
            tokenizer=tokenizer,
            compute_score=final_compute_score,
            **reward_kwargs,
        )
    else:
        return reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=final_compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )


@tqbridge(put_data=False)
def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)
