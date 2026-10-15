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
from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _safe_mean(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return torch.mean(tensor).detach().item()


def _safe_max(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return torch.max(tensor).detach().item()


def _safe_min(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return torch.min(tensor).detach().item()


def _safe_var(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=tensor.device)
    return torch.var(tensor)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()
    original_response_mask = batch.batch.get("original_response_mask", batch.batch["response_mask"]).bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    original_response_length = original_response_mask.sum(dim=-1)
    original_aborted_mask = (original_response_length == 0).bool()
    original_non_aborted_mask = ~original_aborted_mask

    non_aborted_sequence_score = sequence_score[original_non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = _safe_mean(non_aborted_sequence_score)
    score_max = _safe_max(non_aborted_sequence_score)
    score_min = _safe_min(non_aborted_sequence_score)

    reward_mean = _safe_mean(non_aborted_sequence_reward)
    reward_max = _safe_max(non_aborted_sequence_reward)
    reward_min = _safe_min(non_aborted_sequence_reward)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = _safe_var(valid_returns - valid_values)
        return_var = _safe_var(valid_returns)

    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = _safe_mean(non_aborted_response_length)
        non_aborted_response_length_max = _safe_max(non_aborted_response_length)
        non_aborted_response_length_min = _safe_min(non_aborted_response_length)
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        non_aborted_response_length_mean = 0.0
        non_aborted_response_length_max = 0.0
        non_aborted_response_length_min = 0.0
        non_aborted_response_length_clip_ratio = 0.0

    metrics = {
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        "critic/advantages/mean": _safe_mean(valid_adv),
        "critic/advantages/max": _safe_max(valid_adv),
        "critic/advantages/min": _safe_min(valid_adv),
        "critic/returns/mean": _safe_mean(valid_returns),
        "critic/returns/max": _safe_max(valid_returns),
        "critic/returns/min": _safe_min(valid_returns),
        **(
            {
                "critic/values/mean": _safe_mean(valid_values),
                "critic/values/max": _safe_max(valid_values),
                "critic/values/min": _safe_min(valid_values),
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        "response/aborted_ratio": aborted_ratio,
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_variance_proxy_metrics(batch: DataProto, gradient_norm: float = None) -> dict[str, float]:
    metrics = {}

    if "sum_pi_squared" not in batch.batch or "old_log_probs" not in batch.batch or "advantages" not in batch.batch:
        return metrics

    pi_t = torch.exp(batch.batch["old_log_probs"])
    w_per_timestep = 1 - 2 * pi_t + batch.batch["sum_pi_squared"]

    response_mask = batch.batch["response_mask"]

    rollout_is_weights = None
    if "rollout_is_weights" in batch.batch:
        rollout_is_weights = batch.batch["rollout_is_weights"]

        w_per_timestep = w_per_timestep * (rollout_is_weights**2).detach()


    advantages = batch.batch["advantages"]
    advantages_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)

    w_values = verl_F.masked_sum(w_per_timestep, response_mask, axis=-1)


    proxy1_signal_strength = gradient_norm**2 if gradient_norm is not None else None

    if rollout_is_weights is not None:
        rollout_is_weights_scalar = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)
        w_original = verl_F.masked_sum(
            w_per_timestep / torch.clamp((rollout_is_weights**2).detach(), min=1e-10), response_mask, axis=-1
        )
        w_original = torch.clamp(w_original, min=0.0)
        proxy2_total_power = ((rollout_is_weights_scalar**2) * (advantages_scalar**2) * w_original).mean()

    else:
        w_values_clamped = torch.clamp(w_values, min=0.0)
        proxy2_total_power = (advantages_scalar**2 * w_values_clamped).mean()

    proxy3_pure_noise = None
    if proxy1_signal_strength is not None:
        batch_size = advantages_scalar.shape[0]
        if batch_size > 1:
            proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)
            proxy3_pure_noise = max(
                0.0, proxy3_pure_noise.item() if torch.is_tensor(proxy3_pure_noise) else proxy3_pure_noise
            )

    expected_a_squared = (advantages_scalar**2).mean()
    expected_w = w_values.mean()

    metrics.update(
        {
            "variance_proxy/proxy1_signal_strength": (
                proxy1_signal_strength if proxy1_signal_strength is not None else 0.0
            ),
            "variance_proxy/proxy2_total_power": proxy2_total_power.detach().item(),
            "variance_proxy/proxy3_pure_noise": proxy3_pure_noise if proxy3_pure_noise is not None else 0.0,
            "variance_proxy/expected_a_squared": expected_a_squared.detach().item(),
            "variance_proxy/expected_w": expected_w.detach().item(),
        }
    )

    return metrics


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    np.random.seed(seed)
    data_np = np.array(data, dtype=object)
    n_data = len(data_np)

    bootstrap_idxs = np.random.choice(n_data, size=(n_bootstrap, subset_size), replace=True)

    n_fns = len(reduce_fns)
    metric_results = np.empty((n_fns, n_bootstrap), dtype=np.float64)

    for fn_idx, reduce_fn in enumerate(reduce_fns):
        for boot_idx in range(n_bootstrap):
            sample = data_np[bootstrap_idxs[boot_idx]]
            metric_results[fn_idx, boot_idx] = reduce_fn(sample)

    result = [
        (float(np.mean(metric_results[fn_idx])), float(np.std(metric_results[fn_idx]))) for fn_idx in range(n_fns)
    ]
    return result


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    np_mean = np.mean
    np_std = np.std
    reduce_fns_best_worst = [np.max, np.min]
    n_bootstrap = 1000

    def gen_ns(n_resps: int) -> list[int]:
        if n_resps <= 1:
            return []
        ns = []
        n = 2
        while n < n_resps:
            ns.append(n)
            n *= 2
        ns.append(n_resps)
        return ns

    ns_cache = {}

    data_src2uid2var2metric = {}

    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        uid_dict = data_src2uid2var2metric.setdefault(data_source, {})

        for uid, var2vals in uid2var2vals.items():
            pred_vals = var2vals.get("pred")
            has_pred = pred_vals is not None
            var_dict = uid_dict.setdefault(uid, {})

            for var_name, var_vals in var2vals.items():
                if not var_vals or isinstance(var_vals[0], str):
                    continue

                n_resps = len(var_vals)
                metric = {f"mean@{n_resps}": float(np_mean(var_vals))}

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = float(np_std(var_vals))

                    if n_resps not in ns_cache:
                        ns_cache[n_resps] = gen_ns(n_resps)
                    ns = ns_cache[n_resps]

                    for n in ns:
                        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
                            data=var_vals,
                            subset_size=n,
                            reduce_fns=reduce_fns_best_worst,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        metric[f"best@{n}/mean"] = bon_mean
                        metric[f"best@{n}/std"] = bon_std
                        metric[f"worst@{n}/mean"] = won_mean
                        metric[f"worst@{n}/std"] = won_std

                        if has_pred:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, pred_vals, strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                n_bootstrap=n_bootstrap,
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"] = maj_n_mean
                            metric[f"maj@{n}/std"] = maj_n_std

                var_dict[var_name] = metric

    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
    return data_src2var2metric2val
