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
from string import Formatter
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):

    full_logit_distillation: bool = True
    alpha: float = 0.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    enable_reflection: bool = False
    reflection_max_length: int = 1024
    reflection_prompt_max_length: int = 8192
    reflection_truncation: str = "right"
    reflection_model: str = "student"
    reflection_include_for_correct: bool = True
    distill_correct_rollouts: bool = True
    use_error_quote_mask: bool = True
    use_reflection_in_teacher_prompt: bool = True
    skip_distill_if_no_success: bool = True
    skip_distill_if_all_success: bool = True
    reflection_wrong_system_template: str = (
        "You are a careful tutor. Respond strictly in the required format and nothing else. "
        "Do not answer the original question again."
    )
    reflection_wrong_user_template: str = (
        "[Problem]\n"
        "{prompt}\n\n"
        "[Correct Solution]\n\n"
        "{correct_solution}\n\n"
        "[Incorrect Solution]\n\n"
        "{wrong_solution}\n\n"
        "The correct final answer is {answer}. Diagnose the [Incorrect Solution] by responding strictly in the "
        "following format:\n\n"
        "<error_quote>\n"
        "(Extract the EXACT substring from the [Incorrect Solution] where the reasoning first goes wrong)\n"
        "</error_quote>\n"
        "<explaination>\n"
        "(Explain the exact mistake in the quote, how to fix it, and what the correct logic is)\n"
        "</explaination>\n"
    )
    reflection_correct_system_template: str = (
        "You are a careful tutor. Produce exactly one section in the following format and nothing else:\n"
        "<explaination>\n"
        "...\n"
        "</explaination>\n"
        "Do not answer the original question again."
    )
    reflection_correct_user_template: str = (
        "{prompt}\n\n"
        "A correct solution is shown below:\n\n"
        "{correct_solution}\n\n"
        "Summarize why this answer is correct.\n"
    )
    teacher_reflection_template: str = (
        "\n"
        "Reflection:\n\n"
        "{reflection}\n\n"
    )
    teacher_reprompt_template_with_reflection: str = (
        "{prompt}{solution}{feedback}{reflection}\n\n"
        "Correctly solve the original question.\n"
    )
    exp_shock_metrics: bool = False
    exp_shock_thresh: float = 0.5
    exp_shock_hi_quantile: float = 0.99
    debug_print_prompts: bool = False
    debug_print_num_samples: int = 2
    debug_print_max_chars: int = 4000
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False

    def __post_init__(self):
        def _validate_template_fields(template_name: str, template: str, expected_fields: set[str]) -> None:
            used_fields = {
                field_name
                for _, field_name, _, _ in Formatter().parse(template)
                if field_name is not None
            }
            unexpected_fields = used_fields - expected_fields
            if unexpected_fields:
                raise ValueError(
                    f"self_distillation.{template_name} may only use placeholders from {sorted(expected_fields)}, "
                    f"got unexpected placeholders {sorted(unexpected_fields)}"
                )

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        valid_teacher_regularization = ["ema", "trust-region"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        if self.max_reprompt_len <= 0:
            raise ValueError(f"self_distillation.max_reprompt_len must be positive, got {self.max_reprompt_len}")
        if self.reflection_max_length <= 0:
            raise ValueError(
                f"self_distillation.reflection_max_length must be positive, got {self.reflection_max_length}"
            )
        if self.reflection_prompt_max_length <= 0:
            raise ValueError(
                "self_distillation.reflection_prompt_max_length must be positive, "
                f"got {self.reflection_prompt_max_length}"
            )
        if self.debug_print_num_samples <= 0:
            raise ValueError(
                f"self_distillation.debug_print_num_samples must be positive, got {self.debug_print_num_samples}"
            )
        if self.debug_print_max_chars <= 0:
            raise ValueError(
                f"self_distillation.debug_print_max_chars must be positive, got {self.debug_print_max_chars}"
            )
        valid_truncations = ["left", "right", "middle", "error"]
        if self.reprompt_truncation not in valid_truncations:
            raise ValueError(
                f"self_distillation.reprompt_truncation must be one of {valid_truncations}, "
                f"got {self.reprompt_truncation}"
            )
        if self.reflection_truncation not in valid_truncations:
            raise ValueError(
                f"self_distillation.reflection_truncation must be one of {valid_truncations}, "
                f"got {self.reflection_truncation}"
            )
        valid_reflection_models = ["student", "teacher"]
        if self.reflection_model not in valid_reflection_models:
            raise ValueError(
                f"self_distillation.reflection_model must be one of {valid_reflection_models}, "
                f"got {self.reflection_model}"
            )
        _validate_template_fields("reprompt_template", self.reprompt_template, {"prompt", "solution", "feedback"})
        _validate_template_fields("solution_template", self.solution_template, {"successful_previous_attempt"})
        _validate_template_fields("feedback_template", self.feedback_template, {"feedback_raw"})
        _validate_template_fields("reflection_wrong_system_template", self.reflection_wrong_system_template, set())
        _validate_template_fields(
            "reflection_wrong_user_template",
            self.reflection_wrong_user_template,
            {"prompt", "correct_solution", "wrong_solution", "answer"},
        )
        _validate_template_fields("reflection_correct_system_template", self.reflection_correct_system_template, set())
        _validate_template_fields(
            "reflection_correct_user_template",
            self.reflection_correct_user_template,
            {"prompt", "correct_solution", "answer"},
        )
        _validate_template_fields("teacher_reflection_template", self.teacher_reflection_template, {"reflection"})
        _validate_template_fields(
            "teacher_reprompt_template_with_reflection",
            self.teacher_reprompt_template_with_reflection,
            {"prompt", "solution", "feedback", "reflection"},
        )


@dataclass
class RouterReplayConfig(BaseConfig):

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    exp_group_stats: bool = False
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.fsdp_config

        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
