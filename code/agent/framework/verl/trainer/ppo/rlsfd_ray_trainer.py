
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsfd_utils import (
    compute_rlsfd_modulated_advantage,
    compute_rlsfd_role,
)
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


class RLSFDRayTrainer(RayPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rlsfd_cfg = self.config.algorithm.get("rlsfd", {})
        self.rlsfd_inner_lr = float(rlsfd_cfg.get("inner_lr", 1e-5))
        self.rlsfd_teacher_precond = str(rlsfd_cfg.get("teacher_precond", "adamw"))
        self.rlsfd_w_coef = float(rlsfd_cfg.get("w_coef", 0.2))
        self.rlsfd_epsilon_w = float(rlsfd_cfg.get("epsilon_w", 0.2))
        self.rlsfd_mod_center = str(rlsfd_cfg.get("mod_center", "seq"))
        self.rlsfd_no_split = bool(rlsfd_cfg.get("no_split", False))
        self.rlsfd_distill_mode = str(rlsfd_cfg.get("distill_mode", "modulate"))
        print(
            f"[RLSFD] inner_lr={self.rlsfd_inner_lr} teacher_precond={self.rlsfd_teacher_precond} "
            f"w_coef={self.rlsfd_w_coef} epsilon_w={self.rlsfd_epsilon_w} mod_center={self.rlsfd_mod_center} "
            f"no_split={self.rlsfd_no_split} distill_mode={self.rlsfd_distill_mode}"
        )

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="RLSFD Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.rlsfd_no_split:
                        batch.batch["rlsfd_role"] = torch.ones(
                            batch.batch["response_mask"].shape[0], dtype=torch.int64
                        )
                    else:
                        batch.batch["rlsfd_role"] = compute_rlsfd_role(
                            batch.non_tensor_batch["uid"], batch.non_tensor_batch["traj_uid"]
                        )
                    metrics["rlsfd/h1_row_frac"] = float(
                        (batch.batch["rlsfd_role"] == 0).float().mean().item()
                    )

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                    with _timer("teacher_forward", timing_raw):
                        teacher_input = batch.select(
                            batch_keys=[
                                "responses", "input_ids", "attention_mask", "position_ids",
                                "old_log_probs", "advantages", "response_mask", "rlsfd_role",
                            ]
                        )
                        teacher_input.meta_info["rlsfd_inner_lr"] = self.rlsfd_inner_lr
                        teacher_input.meta_info["rlsfd_teacher_precond"] = self.rlsfd_teacher_precond
                        teacher_input.meta_info["rlsfd_no_split"] = self.rlsfd_no_split
                        teacher_output = self.actor_rollout_wg.compute_rlsfd_future_log_prob(teacher_input)
                        batch.batch["teacher_log_probs"] = teacher_output.batch["teacher_log_probs"]
                        inner_metrics = teacher_output.meta_info.get("metrics", {})
                        metrics.update({k: float(v) for k, v in inner_metrics.items()})

                    with _timer("rlsfd_reshape", timing_raw):
                        if self.rlsfd_distill_mode == "opd":
                            with torch.no_grad():
                                rm = batch.batch["response_mask"]
                                gap = (batch.batch["teacher_log_probs"]
                                       - batch.batch["old_log_probs"])
                                denom = rm.sum().clamp(min=1)
                                metrics["rlsfd/gap_abs_mean"] = float(
                                    (gap.abs() * rm).sum() / denom)
                                metrics["rlsfd/gap_mean"] = float(
                                    (gap * rm).sum() / denom)
                                metrics["rlsfd/gap_pos_frac"] = float(
                                    ((gap > 0).float() * rm).sum() / denom)
                                import os as _os
                                if _os.environ.get("DUMP_RLSFD_GAP", "0") == "1" and \
                                        self.global_steps <= int(_os.environ.get("DUMP_RLSFD_GAP_STEPS", "3")):
                                    import json as _json
                                    _dir = _os.environ.get("DUMP_RLSFD_GAP_DIR", "/tmp/rlsfd_gap")
                                    _os.makedirs(_dir, exist_ok=True)
                                    _resp = batch.batch["responses"]
                                    _tlp = batch.batch["teacher_log_probs"]
                                    _slp = batch.batch["old_log_probs"]
                                    _prompts = batch.batch.get("prompts", None)
                                    _nmax = int(_os.environ.get("DUMP_RLSFD_GAP_NSAMPLES", "48"))
                                    _p = _os.path.join(_dir, f"gap_step{self.global_steps}.jsonl")
                                    with open(_p, "w") as _f:
                                        for _i in range(min(_resp.shape[0], _nmax)):
                                            _m = rm[_i].bool()
                                            _rec = {
                                                "token_ids": _resp[_i][_m].to("cpu").tolist(),
                                                "teacher_lp": _tlp[_i][_m].to("cpu").tolist(),
                                                "student_lp": _slp[_i][_m].to("cpu").tolist(),
                                            }
                                            if _prompts is not None:
                                                _pi = _prompts[_i].to("cpu").tolist()
                                                _rec["prompt_ids"] = _pi
                                            _f.write(_json.dumps(_rec) + "\n")
                                    print(f"[RLSFD-DUMP] wrote per-token gap to {_p}")
                        else:
                            token_advantages, rlsfd_metrics = compute_rlsfd_modulated_advantage(
                                seq_advantages=batch.batch["advantages"],
                                student_log_probs=batch.batch["old_log_probs"],
                                teacher_log_probs=batch.batch["teacher_log_probs"],
                                response_mask=batch.batch["response_mask"],
                                role=batch.batch["rlsfd_role"],
                                w_coef=self.rlsfd_w_coef,
                                epsilon_w=self.rlsfd_epsilon_w,
                                mod_center=self.rlsfd_mod_center,
                            )
                            batch.batch["advantages"] = token_advantages
                            metrics.update(rlsfd_metrics)

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
