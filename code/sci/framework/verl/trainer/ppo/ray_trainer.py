# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
import json
import os
import re
import time
import uuid
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from string import Template
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = config.actor_rollout_ref.actor.get("self_distillation", {}).get("reprompt_truncation", "error")
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[:generations_to_log]

        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                if sum_reward:
                    return reward_tensor
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int
    ) -> list[Any]:
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    def _mask_zero_std_reward_groups(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
    ) -> dict[str, float]:
        algorithm_cfg = self.config.algorithm
        if not algorithm_cfg.get("filter_zero_std_groups", False) or "uid" not in batch.non_tensor_batch:
            return {}

        uid_list = batch.non_tensor_batch["uid"]
        if len(uid_list) == 0:
            return {}

        seq_scores = reward_tensor.sum(dim=-1).detach()
        std_eps = float(algorithm_cfg.get("filter_zero_std_groups_eps", 0.0))

        uid_to_indices: dict[Any, list[int]] = {}
        for sample_idx, uid in enumerate(uid_list):
            uid_to_indices.setdefault(uid, []).append(sample_idx)

        masked_group_indices: list[list[int]] = []
        masked_group_stds: list[float] = []

        for indices in uid_to_indices.values():
            if len(indices) <= 1:
                continue

            group_scores = seq_scores[torch.as_tensor(indices, device=seq_scores.device)]
            group_std = torch.std(group_scores).item()
            if group_std <= std_eps:
                masked_group_indices.append(indices)
                masked_group_stds.append(group_std)

        if not masked_group_indices:
            return {}

        mask_indices = [idx for indices in masked_group_indices for idx in indices]
        response_mask = batch.batch["response_mask"]
        response_mask[torch.as_tensor(mask_indices, device=response_mask.device)] = 0

        metrics = {
            "algorithm/zero_std_groups/masked_group_count": float(len(masked_group_indices)),
            "algorithm/zero_std_groups/kept_group_count": float(len(uid_to_indices) - len(masked_group_indices)),
            "algorithm/zero_std_groups/masked_sample_count": float(len(mask_indices)),
            "algorithm/zero_std_groups/kept_sample_count": float(len(batch) - len(mask_indices)),
            "algorithm/zero_std_groups/masked_group_fraction": float(len(masked_group_indices) / len(uid_to_indices)),
            "algorithm/zero_std_groups/std_threshold": std_eps,
            "algorithm/zero_std_groups/max_masked_std": max(masked_group_stds) if masked_group_stds else 0.0,
            "algorithm/zero_std_groups/final_group_count": float(len(uid_to_indices)),
            "algorithm/zero_std_groups/final_sample_count": float(len(batch)),
        }
        if len(masked_group_indices) == len(uid_to_indices):
            metrics["algorithm/zero_std_groups/all_groups_masked"] = 1.0
        return metrics

    def _collect_solutions_by_uid(self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    @staticmethod
    def _extract_reflection_text(text: str) -> str:
        match = re.search(r"<(?:explaination|explanation|reflection)>\s*(.*?)\s*</(?:explaination|explanation|reflection)>", text, flags=re.DOTALL | re.IGNORECASE)
        if match is not None:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _extract_tag_content(text: str, tag_name: str) -> Optional[str]:
        match = re.search(
            rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            return None
        return match.group(1)

    def _extract_reflection_fields(self, text: str) -> tuple[Optional[str], Optional[str]]:
        reflection_text = (
            self._extract_tag_content(text, "explaination")
            or self._extract_tag_content(text, "explanation")
            or self._extract_tag_content(text, "reflection")
        )
        error_quote = self._extract_tag_content(text, "error_quote")

        if reflection_text is not None:
            reflection_text = reflection_text.strip()
        if error_quote is not None and error_quote.strip() == "":
            error_quote = None

        return reflection_text or None, error_quote

    def _find_first_error_token_index(
        self,
        response_text: str,
        error_quote: Optional[str],
        response_length: int,
    ) -> tuple[int, bool]:
        if not error_quote:
            return 0, False

        candidates = [error_quote]
        stripped_quote = error_quote.strip()
        if stripped_quote and stripped_quote != error_quote:
            candidates.append(stripped_quote)

        for candidate in candidates:
            start_char = response_text.find(candidate)
            if start_char < 0:
                continue
            prefix_token_ids = self.tokenizer.encode(
                response_text[:start_char],
                add_special_tokens=False,
            )
            return min(len(prefix_token_ids), response_length), True

        return 0, False

    def _build_self_distillation_token_mask(
        self,
        response_texts: list[str],
        response_mask: torch.Tensor,
        error_quotes: list[Optional[str]],
        reflection_wrong_flags: list[bool],
        use_error_quote_mask: bool = True,
    ) -> tuple[torch.Tensor, list[bool], list[int], list[float]]:
        token_mask = torch.ones_like(response_mask, dtype=torch.float32)
        error_quote_found_flags = [False] * len(response_texts)
        error_quote_start_token_indices = [0] * len(response_texts)
        error_quote_start_token_fractions = [0.0] * len(response_texts)

        if not use_error_quote_mask:
            return token_mask, error_quote_found_flags, error_quote_start_token_indices, error_quote_start_token_fractions

        for i, response_text in enumerate(response_texts):
            if not reflection_wrong_flags[i]:
                continue

            response_length = int(response_mask[i].sum().item())
            token_index, found = self._find_first_error_token_index(
                response_text=response_text,
                error_quote=error_quotes[i],
                response_length=response_length,
            )
            error_quote_found_flags[i] = found
            error_quote_start_token_indices[i] = token_index
            if response_length > 0:
                error_quote_start_token_fractions[i] = token_index / response_length

            if token_index > 0:
                token_mask[i, :token_index] = 0.0

        return token_mask, error_quote_found_flags, error_quote_start_token_indices, error_quote_start_token_fractions

    @staticmethod
    def _truncate_debug_text(text: Optional[str], max_chars: int) -> str:
        if text is None:
            return "<none>"
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...<truncated>"

    def _maybe_debug_print_self_distillation_samples(
        self,
        prompt_texts: list[str],
        response_texts: list[str],
        solution_strs: list[Optional[str]],
        reflection_texts: list[Optional[str]],
        error_quotes: list[Optional[str]],
        raw_reflection_outputs: list[Optional[str]],
        teacher_messages: list[list[dict]],
        self_distillation_mask: torch.Tensor,
        self_distillation_cfg,
    ) -> None:
        if not self_distillation_cfg.debug_print_prompts:
            return

        max_samples = min(self_distillation_cfg.debug_print_num_samples, len(prompt_texts))
        max_chars = self_distillation_cfg.debug_print_max_chars
        printed = 0
        for i in range(len(prompt_texts)):
            if printed >= max_samples:
                break
            teacher_prompt_text = self.tokenizer.apply_chat_template(
                [teacher_messages[i]],
                tokenize=False,
                continue_final_message=False,
                add_generation_prompt=True,
            )
            print("=" * 80)
            print(f"[self_distillation debug] sample_index={i} mask={float(self_distillation_mask[i].item())}")
            print("[reflector_output]")
            print(self._truncate_debug_text(raw_reflection_outputs[i], max_chars))
            print("[teacher_prompt]")
            print(self._truncate_debug_text(teacher_prompt_text, max_chars))
            printed += 1

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
    ) -> Optional[str]:
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]
        if len(solution_idxs) == 0:
            return None
        solution_idx = solution_idxs[0]
        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)
        return solution_str

    def _build_reflection_messages(
        self,
        prompt_texts: list[str],
        response_texts: list[str],
        solution_strs: list[Optional[str]],
        answer_texts: list[str],
        sample_success_flags: list[bool],
        group_has_success_flags: list[bool],
        group_all_success_flags: list[bool],
        self_distillation_cfg,
    ) -> tuple[list[Optional[list[dict]]], list[bool], list[bool]]:
        reflection_messages: list[Optional[list[dict]]] = [None] * len(prompt_texts)
        reflection_wrong_flags = [False] * len(prompt_texts)
        reflection_correct_flags = [False] * len(prompt_texts)
        distill_correct_rollouts = self_distillation_cfg.get("distill_correct_rollouts", True)

        for i in range(len(prompt_texts)):
            if self_distillation_cfg.skip_distill_if_no_success and not group_has_success_flags[i]:
                continue
            if self_distillation_cfg.skip_distill_if_all_success and group_all_success_flags[i]:
                continue

            if sample_success_flags[i]:
                if not distill_correct_rollouts:
                    continue
                if not self_distillation_cfg.reflection_include_for_correct:
                    continue
                reflection_prompt = self_distillation_cfg.reflection_correct_user_template.format(
                    prompt=prompt_texts[i],
                    correct_solution=response_texts[i],
                    answer=answer_texts[i],
                )
                reflection_messages[i] = [
                    {"role": "system", "content": self_distillation_cfg.reflection_correct_system_template},
                    {"role": "user", "content": reflection_prompt},
                ]
                reflection_correct_flags[i] = True
            elif solution_strs[i] is not None:
                reflection_prompt = self_distillation_cfg.reflection_wrong_user_template.format(
                    prompt=prompt_texts[i],
                    correct_solution=solution_strs[i],
                    wrong_solution=response_texts[i],
                    answer=answer_texts[i],
                )
                reflection_messages[i] = [
                    {"role": "system", "content": self_distillation_cfg.reflection_wrong_system_template},
                    {"role": "user", "content": reflection_prompt},
                ]
                reflection_wrong_flags[i] = True

        return reflection_messages, reflection_wrong_flags, reflection_correct_flags

    def _truncate_reflection_message_for_rollout(
        self,
        message: list[dict],
        max_prompt_length: int,
    ) -> list[dict]:
        if not self.async_rollout_mode:
            return message

        truncated_message = [dict(item) for item in message]
        user_content = truncated_message[-1]["content"]
        user_token_ids = self.tokenizer.encode(user_content, add_special_tokens=False)
        if not user_token_ids:
            return truncated_message

        while True:
            prompt_ids = self.tokenizer.apply_chat_template(
                truncated_message,
                tokenize=True,
                add_generation_prompt=True,
                **(self.config.data.apply_chat_template_kwargs or {}),
            )
            prompt_len = len(prompt_ids)
            if prompt_len <= max_prompt_length:
                return truncated_message

            overflow = prompt_len - max_prompt_length
            new_len = max(1, len(user_token_ids) - overflow - 8)
            if new_len >= len(user_token_ids):
                new_len = len(user_token_ids) - 1
            user_token_ids = user_token_ids[:new_len]
            truncated_message[-1]["content"] = self.tokenizer.decode(
                user_token_ids,
                skip_special_tokens=True,
            )
            if len(user_token_ids) <= 1:
                return truncated_message

    def _generate_reflections(
        self,
        reflection_messages: list[Optional[list[dict]]],
        self_distillation_cfg,
    ) -> tuple[list[Optional[str]], list[Optional[str]], list[Optional[str]], dict[str, float]]:
        reflection_texts: list[Optional[str]] = [None] * len(reflection_messages)
        error_quotes: list[Optional[str]] = [None] * len(reflection_messages)
        raw_reflection_outputs: list[Optional[str]] = [None] * len(reflection_messages)
        metrics = {
            "self_distillation/reflection_prompt_length_mean": 0.0,
            "self_distillation/reflection_prompt_length_max": 0.0,
            "self_distillation/reflection_output_length_mean": 0.0,
            "self_distillation/reflection_output_length_max": 0.0,
        }
        valid_indices = [i for i, message in enumerate(reflection_messages) if message is not None]
        if not valid_indices:
            return reflection_texts, error_quotes, raw_reflection_outputs, metrics

        valid_messages = [reflection_messages[i] for i in valid_indices]
        valid_messages = [
            self._truncate_reflection_message_for_rollout(
                message=message,
                max_prompt_length=self.config.actor_rollout_ref.rollout.prompt_length,
            )
            for message in valid_messages
        ]
        enable_thinking = (
            self.config.data.apply_chat_template_kwargs.get("enable_thinking", True)
            if self.config.data.apply_chat_template_kwargs
            else True
        )
        tokenized = self.tokenizer.apply_chat_template(
            valid_messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            padding=True,
            truncation=False,
        )
        input_ids, attention_mask = postprocess_data(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_length=self_distillation_cfg.reflection_prompt_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self_distillation_cfg.reflection_truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        prompt_lengths = attention_mask.sum(dim=-1).to(dtype=torch.float32)

        reflection_batch = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors={
                "raw_prompt": np.array(valid_messages, dtype=object),
            },
            meta_info={
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "response_length": self_distillation_cfg.reflection_max_length,
                "global_steps": self.global_steps,
            },
        )
        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        reflection_batch_padded, pad_size = pad_dataproto_to_divisor(reflection_batch, size_divisor)
        use_teacher_reflector = getattr(self_distillation_cfg, "reflection_model", "student") == "teacher"
        if not self.async_rollout_mode:
            if use_teacher_reflector:
                reflection_output_padded = self.actor_rollout_wg.generate_sequences_with_teacher(reflection_batch_padded)
            else:
                reflection_output_padded = self.actor_rollout_wg.generate_sequences(reflection_batch_padded)
        else:
            if use_teacher_reflector:
                warnings.warn(
                    "reflection_model=teacher is not supported in async_rollout_mode; falling back to student.",
                    stacklevel=2,
                )
            reflection_output_padded = self.async_rollout_manager.generate_sequences(reflection_batch_padded)
        reflection_output = unpad_dataproto(reflection_output_padded, pad_size=pad_size)

        for output_idx, batch_idx in enumerate(valid_indices):
            reflection_output_text = self.tokenizer.decode(
                reflection_output.batch["responses"][output_idx],
                skip_special_tokens=True,
            ).strip()
            raw_reflection_outputs[batch_idx] = reflection_output_text or None
            reflection_text, error_quote = self._extract_reflection_fields(reflection_output_text)
            reflection_texts[batch_idx] = reflection_text
            error_quotes[batch_idx] = error_quote
        response_lengths = reflection_output.batch["response_mask"].sum(dim=-1).to(dtype=torch.float32)

        metrics = {
            "self_distillation/reflection_prompt_length_mean": prompt_lengths.mean().item(),
            "self_distillation/reflection_prompt_length_max": prompt_lengths.max().item(),
            "self_distillation/reflection_output_length_mean": response_lengths.mean().item(),
            "self_distillation/reflection_output_length_max": response_lengths.max().item(),
            "self_distillation/reflection_model": getattr(self_distillation_cfg, "reflection_model", "student"),
        }
        return reflection_texts, error_quotes, raw_reflection_outputs, metrics

    def _build_teacher_message(
        self,
        system_messages: list[dict],
        prompt_text: str,
        solution_str: Optional[str],
        feedback_text: Optional[str],
        reflection_text: Optional[str],
        self_distillation_cfg,
    ) -> list[dict]:
        has_solution = solution_str is not None
        has_feedback = feedback_text is not None
        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

        solution_section = ""
        if has_solution:
            solution_section = self_distillation_cfg.solution_template.format(
                successful_previous_attempt=solution_str
            )

        feedback_section = ""
        if use_feedback:
            feedback_section = self_distillation_cfg.feedback_template.format(
                feedback_raw=feedback_text
            )

        reflection_section = ""
        if self_distillation_cfg.get("use_reflection_in_teacher_prompt", True) and reflection_text is not None:
            reflection_section = self_distillation_cfg.teacher_reflection_template.format(
                reflection=reflection_text
            )

        if reflection_section:
            reprompt_text = self_distillation_cfg.teacher_reprompt_template_with_reflection.format(
                prompt=prompt_text,
                solution=solution_section,
                feedback=feedback_section,
                reflection=reflection_section,
            )
        elif use_feedback or has_solution:
            reprompt_text = self_distillation_cfg.reprompt_template.format(
                prompt=prompt_text,
                solution=solution_section,
                feedback=feedback_section,
            )
        else:
            reprompt_text = prompt_text

        return system_messages + [{"role": "user", "content": reprompt_text}]

    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode != "sdpo":
            return None

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        system_prompts = [list(msgs[:-1]) for msgs in batch.non_tensor_batch["raw_prompt"]]
        answer_texts = [
            str(item.non_tensor_batch.get("reward_model", {}).get("ground_truth", ""))
            for item in batch
        ]
        batch_size = batch.batch.batch_size[0]

        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, success_reward_threshold=self_distillation_cfg.success_reward_threshold)
        sample_success_flags = [seq_scores[i] >= self_distillation_cfg.success_reward_threshold for i in range(batch_size)]
        uid_to_group_size = Counter(batch.non_tensor_batch["uid"])
        group_has_success_flags = [len(success_by_uid[uid]) > 0 for uid in batch.non_tensor_batch["uid"]]
        group_all_success_flags = [
            len(success_by_uid[uid]) == uid_to_group_size[uid]
            for uid in batch.non_tensor_batch["uid"]
        ]
        skip_distill_flags = [
            (
                self_distillation_cfg.skip_distill_if_no_success
                and not group_has_success_flags[i]
            )
            or (
                self_distillation_cfg.skip_distill_if_all_success
                and group_all_success_flags[i]
            )
            for i in range(batch_size)
        ]
        solution_strs = [
            self._get_solution(
                i,
                success_by_uid,
                batch.non_tensor_batch["uid"],
                response_texts,
                self_distillation_cfg.dont_reprompt_on_self_success,
                self_distillation_cfg.get("remove_thinking_from_demonstration", False),
            )
            for i in range(batch_size)
        ]
        reflection_texts: list[Optional[str]] = [None] * batch_size
        error_quotes: list[Optional[str]] = [None] * batch_size
        raw_reflection_outputs: list[Optional[str]] = [None] * batch_size
        reflection_wrong_flags = [False] * batch_size
        reflection_correct_flags = [False] * batch_size
        reflection_metrics = {
            "self_distillation/reflection_prompt_length_mean": 0.0,
            "self_distillation/reflection_prompt_length_max": 0.0,
            "self_distillation/reflection_output_length_mean": 0.0,
            "self_distillation/reflection_output_length_max": 0.0,
        }
        if self_distillation_cfg.enable_reflection:
            reflection_messages, reflection_wrong_flags, reflection_correct_flags = self._build_reflection_messages(
                prompt_texts=prompt_texts,
                response_texts=response_texts,
                solution_strs=solution_strs,
                answer_texts=answer_texts,
                sample_success_flags=sample_success_flags,
                group_has_success_flags=group_has_success_flags,
                group_all_success_flags=group_all_success_flags,
                self_distillation_cfg=self_distillation_cfg,
            )
            reflection_texts, error_quotes, raw_reflection_outputs, reflection_metrics = self._generate_reflections(
                reflection_messages,
                self_distillation_cfg,
            )

        messages = [
            self._build_teacher_message(
                system_messages=system_prompts[i],
                prompt_text=prompt_texts[i],
                solution_str=solution_strs[i],
                feedback_text=feedback_list[i],
                reflection_text=reflection_texts[i],
                self_distillation_cfg=self_distillation_cfg,
            )
            for i in range(batch_size)
        ]
        enable_thinking = self.config.data.apply_chat_template_kwargs.get("enable_thinking", True) if self.config.data.apply_chat_template_kwargs else True
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=self_distillation_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]
        reflection_used = [reflection_texts[i] is not None for i in range(batch_size)]
        reflection_used_in_teacher = [
            self_distillation_cfg.get("use_reflection_in_teacher_prompt", True) and reflection_used[i]
            for i in range(batch_size)
        ]
        distill_correct_rollouts = self_distillation_cfg.get("distill_correct_rollouts", True)

        self_distillation_mask = torch.tensor(
            [
                (
                    solution_strs[i] is not None or feedback_used[i] or reflection_used_in_teacher[i]
                )
                and not skip_distill_flags[i]
                and (distill_correct_rollouts or not sample_success_flags[i])
                for i in range(batch_size)
            ],
            dtype=torch.float32,
            device=device
        )
        (
            self_distillation_token_mask,
            error_quote_found_flags,
            error_quote_start_token_indices,
            error_quote_start_token_fractions,
        ) = self._build_self_distillation_token_mask(
            response_texts=response_texts,
            response_mask=response_mask,
            error_quotes=error_quotes,
            reflection_wrong_flags=reflection_wrong_flags,
            use_error_quote_mask=self_distillation_cfg.get("use_error_quote_mask", True),
        )
        self_distillation_token_mask = self_distillation_token_mask.to(device=device)
        self._maybe_debug_print_self_distillation_samples(
            prompt_texts=prompt_texts,
            response_texts=response_texts,
            solution_strs=solution_strs,
            reflection_texts=reflection_texts,
            error_quotes=error_quotes,
            raw_reflection_outputs=raw_reflection_outputs,
            teacher_messages=messages,
            self_distillation_mask=self_distillation_mask,
            self_distillation_cfg=self_distillation_cfg,
        )

        uids = set(batch.non_tensor_batch["uid"])
        no_success_group_fraction = len([uid for uid in uids if len(success_by_uid[uid]) == 0]) / len(uids)
        all_success_group_fraction = len([uid for uid in uids if len(success_by_uid[uid]) == uid_to_group_size[uid]]) / len(uids)
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)
        num_with_reflection = sum(1 for reflection in reflection_texts if reflection is not None)
        wrong_sample_count = sum(reflection_wrong_flags)
        found_start_token_indices = [
            error_quote_start_token_indices[i]
            for i in range(batch_size)
            if error_quote_found_flags[i]
        ]
        found_start_token_fractions = [
            error_quote_start_token_fractions[i]
            for i in range(batch_size)
            if error_quote_found_flags[i]
        ]
        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
            "self_distillation/reflection_enabled": float(self_distillation_cfg.enable_reflection),
            "self_distillation/distill_correct_rollouts": float(distill_correct_rollouts),
            "self_distillation/use_error_quote_mask": float(self_distillation_cfg.get("use_error_quote_mask", True)),
            "self_distillation/use_reflection_in_teacher_prompt": float(
                self_distillation_cfg.get("use_reflection_in_teacher_prompt", True)
            ),
            "self_distillation/reflection_generated_fraction": num_with_reflection / batch_size,
            "self_distillation/reflection_for_wrong_fraction": sum(reflection_wrong_flags) / batch_size,
            "self_distillation/reflection_for_correct_fraction": sum(reflection_correct_flags) / batch_size,
            "self_distillation/no_success_group_fraction": no_success_group_fraction,
            "self_distillation/skipped_group_fraction_due_to_no_success": (
                no_success_group_fraction if self_distillation_cfg.skip_distill_if_no_success else 0.0
            ),
            "self_distillation/all_success_group_fraction": all_success_group_fraction,
            "self_distillation/skipped_group_fraction_due_to_all_success": (
                all_success_group_fraction if self_distillation_cfg.skip_distill_if_all_success else 0.0
            ),
            "self_distillation/teacher_prompt_with_reflection_fraction": sum(reflection_used_in_teacher) / batch_size,
            "self_distillation/error_quote_found_fraction": sum(error_quote_found_flags) / batch_size,
            "self_distillation/error_quote_found_among_wrong_fraction": (
                sum(error_quote_found_flags) / max(1, wrong_sample_count)
            ),
            "self_distillation/error_quote_start_token_mean": (
                sum(error_quote_start_token_indices) / max(1, wrong_sample_count)
            ),
            "self_distillation/error_quote_start_token_fraction_mean": (
                sum(error_quote_start_token_fractions) / max(1, wrong_sample_count)
            ),
            "self_distillation/error_quote_start_token_among_found_mean": (
                sum(found_start_token_indices) / max(1, len(found_start_token_indices))
            ),
            "self_distillation/error_quote_start_token_fraction_among_found_mean": (
                sum(found_start_token_fractions) / max(1, len(found_start_token_fractions))
            ),
            "self_distillation/suffix_target_token_fraction": (
                (self_distillation_token_mask * self_distillation_mask.unsqueeze(1)).sum().item()
                / max(1.0, (response_mask * self_distillation_mask.unsqueeze(1)).sum().item())
            ),
        }
        metrics.update(reflection_metrics)
        return DataProto.from_dict(tensors={
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
            "self_distillation_token_mask": self_distillation_token_mask,
        }), metrics

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"}) & batch.non_tensor_batch.keys()

        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            batch_size = len(scores)
            prev_num_scores = len(sample_scores)
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})

            def _default_val_for_reward_key(key: str):
                if key in {"pred", "feedback"}:
                    return ""
                return 0

            current_batch_infos = {}
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    current_batch_infos[key] = values.tolist()
                else:
                    current_batch_infos[key] = values if isinstance(values, list) else [values]

            for key, values in current_batch_infos.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = [_default_val_for_reward_key(key)] * prev_num_scores
                reward_extra_infos_dict[key].extend(values)

            for key in list(reward_extra_infos_dict.keys()):
                if key in {"reward"} or key in current_batch_infos:
                    continue
                reward_extra_infos_dict[key].extend([_default_val_for_reward_key(key)] * batch_size)

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        def _default_val_for_reward_key(key: str):
            if key in {"pred", "feedback"}:
                return ""
            return 0

        for key_info, lst in reward_extra_infos_dict.items():
            if len(lst) < len(sample_scores):
                lst.extend([_default_val_for_reward_key(key_info)] * (len(sample_scores) - len(lst)))
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        if not self.use_reward_loop:
            if self.use_rm:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        self.async_rollout_mode = True

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            log_probs = tu.get(output, "log_probs")
            log_probs = no_padding_2_padding(log_probs, batch_td)
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            group_name=self.config.trainer.get("group_name", None),
        )

        self.global_steps = 0

        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["original_response_mask"] = batch.batch["response_mask"].clone()
                    zero_std_group_metrics = self._mask_zero_std_reward_groups(batch=batch, reward_tensor=reward_tensor)
                    metrics.update(zero_std_group_metrics)
                    batch.batch["token_level_scores"] = reward_tensor

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        self_distillation_data = self._maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra_infos_dict)
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
