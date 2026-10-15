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
import itertools
import time
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

            else:
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.get("use_sdl_loss", False) or self.config.get("use_sdar_loss", False):
            select_keys.append("teacher_log_probs")
        if self.config.get("use_sf_loss", False) and "sf_teacher_log_probs" in data.batch.keys():
            select_keys.append("sf_teacher_log_probs")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.get("use_sdl_loss", False):
                        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdl_loss = compute_sdl_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdl_coef = self.config.get("sdl_loss_coef", 0.1)
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        metrics["actor/sdl_loss"] = sdl_loss.detach().item()
                        metrics["actor/sdl_coef"] = sdl_coef

                    if self.config.get("use_sdar_loss", False):
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        teacher_log_probs = data["teacher_log_probs"]
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                            positive_only=self.config.get("sdar_positive_only", False),
                            opsd_uniform=self.config.get("sdar_opsd_uniform", False),
                            cut_negative=self.config.get("sdar_cut_negative", False),
                            empty_guard=self.config.get("sdar_empty_guard", False),
                        )
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        metrics.update(sdar_metrics)
                        metrics["sdar/coef"] = sdar_coef

                    if self.config.get("use_sf_loss", False) and "sf_teacher_log_probs" in data.keys():
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        sf_loss, sf_raw_metrics = compute_sdar_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=data["sf_teacher_log_probs"],
                            response_mask=response_mask,
                            gate_beta=self.config.get("sf_gate_beta", 0.0),
                            loss_agg_mode=loss_agg_mode,
                            positive_only=False,
                            opsd_uniform=False,
                            cut_negative=False,
                            empty_guard=self.config.get("sdar_empty_guard", False),
                        )
                        sf_coef = self.config.get("sf_loss_coef", 0.01)
                        policy_loss = policy_loss + sf_loss * sf_coef
                        metrics.update({k.replace("sdar/", "sf/", 1) if k.startswith("sdar/") else ("sf/" + k): v
                                        for k, v in sf_raw_metrics.items()})
                        metrics["sf/coef"] = sf_coef


                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_rlsfd_future_log_prob(self, data: DataProto):
        temperature = data.meta_info["temperature"]
        micro_bsz = int(data.meta_info.get("micro_batch_size", self.config.ppo_micro_batch_size_per_gpu))
        inner_lr = float(data.meta_info.get("rlsfd_inner_lr", 1e-5))
        precond = str(data.meta_info.get("rlsfd_teacher_precond", "adamw"))
        no_split = bool(data.meta_info.get("rlsfd_no_split", False))
        _kappa_raw = data.meta_info.get("rlsfd_kappa", None)
        kappa = None if _kappa_raw is None else float(_kappa_raw)
        _tgt_raw = data.meta_info.get("rlsfd_target_role", None)
        target_role = None if _tgt_raw is None else int(_tgt_raw)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids",
                       "old_log_probs", "advantages", "response_mask", "rlsfd_role"]
        teacher_keys = ["teacher_input_ids", "teacher_attention_mask", "teacher_position_ids"]
        has_teacher_inputs = all(k in data.batch.keys() for k in teacher_keys)
        if has_teacher_inputs:
            select_keys = select_keys + teacher_keys
        batch_all = data.select(batch_keys=select_keys).batch
        device = get_torch_device().current_device()

        def _sync_max(n: int) -> int:
            if torch.distributed.is_initialized():
                t = torch.tensor([n], device=device, dtype=torch.long)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
                n = int(t.cpu().item())
            return n

        self.actor_module.train()
        self.actor_optimizer.zero_grad()

        role = batch_all["rlsfd_role"]
        if no_split:
            h1_idx = torch.arange(role.numel(), device=role.device)
        else:
            h1_idx = torch.nonzero(role == 0, as_tuple=True)[0]
        real_h1 = int(h1_idx.numel())
        phase_a_all = batch_all
        if has_teacher_inputs:
            phase_a_all = batch_all.clone()
            for k in teacher_keys:
                del phase_a_all[k]
        if real_h1 > 0:
            sub = phase_a_all[h1_idx]
        else:
            sub = phase_a_all[:1].clone()
            sub["advantages"] = torch.zeros_like(sub["advantages"])

        num_micro = _sync_max(max(1, -(-sub.batch_size[0] // micro_bsz)))
        _dbg_num_micro, _dbg_micro_bsz = float(num_micro), float(micro_bsz)
        need_rows = num_micro * micro_bsz
        if sub.batch_size[0] < need_rows:
            reps = torch.tensor(
                [i % sub.batch_size[0] for i in range(need_rows - sub.batch_size[0])], dtype=torch.long)
            pad = sub[reps].clone()
            pad["advantages"] = torch.zeros_like(pad["advantages"])
            sub = torch.cat([sub, pad], dim=0)

        total_local = max(real_h1, 1)
        inner_pg_loss = 0.0
        clip_ratio = self.config.clip_ratio
        for mb in sub.split(micro_bsz):
            mb = mb.to(device)
            response_mask = mb["response_mask"]
            _, log_prob = self._forward_micro_batch(
                {k: v for k, v in mb.items()}, temperature=temperature, calculate_entropy=False)
            pg_loss, _, _, _ = compute_policy_loss(
                old_log_prob=mb["old_log_probs"],
                log_prob=log_prob,
                advantages=mb["advantages"],
                response_mask=response_mask,
                cliprange=clip_ratio,
                cliprange_low=clip_ratio,
                cliprange_high=clip_ratio,
                clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                loss_agg_mode=self.config.loss_agg_mode,
            )
            scale = response_mask.shape[0] / total_local
            (pg_loss * scale).backward()
            inner_pg_loss += pg_loss.detach().item() * scale

        if isinstance(self.actor_module, FSDP):
            inner_grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            inner_grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            inner_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        grad_finite = bool(torch.isfinite(inner_grad_norm))

        pg_of = {}
        if precond == "adamw":
            for grp in self.actor_optimizer.param_groups:
                for p in grp["params"]:
                    pg_of[id(p)] = grp

        saved = []
        shadow_delta_sq = 0.0
        with torch.no_grad():
            for param in self.actor_module.parameters():
                if not (grad_finite and param.grad is not None):
                    continue
                g32 = param.grad.to(torch.float32)
                if precond == "adamw":
                    grp = pg_of.get(id(param), self.actor_optimizer.param_groups[0])
                    beta1, beta2 = grp.get("betas", (0.9, 0.999))
                    eps_a = grp.get("eps", 1e-8)
                    st = self.actor_optimizer.state.get(param, {})
                    m = st.get("exp_avg")
                    v = st.get("exp_avg_sq")
                    step_t = st.get("step", None)
                    t = float(step_t.item()) if torch.is_tensor(step_t) else float(step_t or 0.0)
                    m32 = m.to(torch.float32) if m is not None else torch.zeros_like(g32)
                    v32 = v.to(torch.float32) if v is not None else torch.zeros_like(g32)
                    if kappa is None:
                        m_new = beta1 * m32 + (1.0 - beta1) * g32
                    else:
                        m_new = (1.0 - kappa) * m32 + kappa * g32
                    v_new = beta2 * v32 + (1.0 - beta2) * g32 * g32
                    bc1 = 1.0 - beta1 ** (t + 1.0)
                    bc2 = 1.0 - beta2 ** (t + 1.0)
                    update = (m_new / bc1) / ((v_new / bc2).sqrt() + eps_a)
                else:
                    update = g32
                orig = param.data.clone()
                saved.append((param, orig))
                future = param.data.to(torch.float32) - inner_lr * update
                param.data.copy_(future.to(param.data.dtype))
                shadow_delta_sq += float((param.data - orig).to(torch.float32).pow(2).sum().item())

        self.actor_module.eval()
        try:
            if has_teacher_inputs:
                fwd = batch_all.clone()
                fwd["input_ids"] = batch_all["teacher_input_ids"]
                fwd["attention_mask"] = batch_all["teacher_attention_mask"]
                fwd["position_ids"] = batch_all["teacher_position_ids"]
                for k in ("teacher_input_ids", "teacher_attention_mask", "teacher_position_ids"):
                    del fwd[k]
            else:
                fwd = batch_all
            full_rows = fwd.batch_size[0]
            tgt_idx = None
            if target_role is not None:
                tgt_idx = torch.nonzero(role == target_role, as_tuple=True)[0]
                if int(tgt_idx.numel()) > 0:
                    fwd = fwd[tgt_idx]
                else:
                    fwd = fwd[:1].clone()
            local_rows = fwd.batch_size[0]
            num_micro_f = _sync_max(max(1, -(-local_rows // micro_bsz)))
            need_rows_f = num_micro_f * micro_bsz
            if local_rows < need_rows_f:
                reps = torch.tensor(
                    [i % local_rows for i in range(need_rows_f - local_rows)], dtype=torch.long)
                fwd = torch.cat([fwd, fwd[reps]], dim=0)
            teacher_lp_lst = []
            for mb in fwd.split(micro_bsz):
                mb = mb.to(device)
                with torch.no_grad():
                    _, lp = self._forward_micro_batch(
                        {k: v for k, v in mb.items()}, temperature=temperature, calculate_entropy=False)
                teacher_lp_lst.append(lp)
            teacher_log_probs = torch.cat(teacher_lp_lst, dim=0)[:local_rows]
            if target_role is not None:
                scattered = torch.full(
                    (full_rows, teacher_log_probs.shape[-1]),
                    float("nan"), dtype=teacher_log_probs.dtype, device=teacher_log_probs.device)
                if tgt_idx is not None and int(tgt_idx.numel()) > 0:
                    scattered[tgt_idx.to(scattered.device)] = teacher_log_probs
                teacher_log_probs = scattered
        finally:
            with torch.no_grad():
                for param, orig in saved:
                    param.data.copy_(orig)
            saved.clear()
            self.actor_optimizer.zero_grad()

        metrics = {
            "rlsfd/inner_pg_loss": inner_pg_loss,
            "rlsfd/inner_grad_norm": float(inner_grad_norm.detach().item()),
            "rlsfd/inner_grad_finite": float(grad_finite),
            "rlsfd/inner_lr": inner_lr,
            "rlsfd/shadow_delta_l2_localshard": shadow_delta_sq ** 0.5,
            "rlsfd/num_micro": _dbg_num_micro,
            "rlsfd/micro_bsz": _dbg_micro_bsz,
            "rlsfd/h1_rows_local": float(real_h1),
        }
        return teacher_log_probs, metrics
