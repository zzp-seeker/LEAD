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
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig", "RolloutCorrectionConfig"]


@dataclass
class KLControlConfig(BaseConfig):

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class RolloutCorrectionConfig(BaseConfig):

    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: float = 2.0
    rollout_is_batch_normalize: bool = False
    rollout_rs: Optional[str] = None
    rollout_rs_threshold: Optional[str | float] = None
    bypass_mode: bool = False
    loss_type: str = "ppo_clip"

    @classmethod
    def decoupled_token_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        return cls(rollout_is="token", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        return cls(rollout_is="sequence", rollout_is_threshold=threshold, rollout_rs=None)

    @classmethod
    def decoupled_seq_is_rs(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.5_2.0",
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_sum_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def bypass_ppo_clip(cls) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is=None,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_ppo_clip_k3_rs(
        cls,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="ppo_clip",
        )

    @classmethod
    def bypass_pg_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=threshold,
            rollout_rs=None,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs(
        cls,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def bypass_pg_geo_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def bypass_pg_geo_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: Optional[str | float] = "0.999_1.001",
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold=rs_threshold,
            bypass_mode=True,
            loss_type="reinforce",
        )

    @classmethod
    def decoupled_k3_rs(
        cls,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is=None,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_k3_rs_seq_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is="sequence",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def decoupled_k3_rs_token_tis(
        cls,
        is_threshold: float = 2.0,
        rs_threshold: float = 0.01,
    ) -> "RolloutCorrectionConfig":
        return cls(
            rollout_is="token",
            rollout_is_threshold=is_threshold,
            rollout_rs="seq_mean_k3",
            rollout_rs_threshold=rs_threshold,
        )

    @classmethod
    def disabled(cls) -> "RolloutCorrectionConfig":
        return cls(rollout_is=None, rollout_rs=None)


@dataclass
class AlgoConfig(BaseConfig):

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    filter_zero_std_groups: bool = False
    filter_zero_std_groups_eps: float = 0.0
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None
    rollout_correction: Optional[RolloutCorrectionConfig] = None
