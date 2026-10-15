
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
from verl.trainer.ppo.a7_utils import (
    anneal_rho,
    compute_a7_twoside_advantage,
    make_crossfit_folds,
)
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


class A7RayTrainer(RayPPOTrainer):

    def __init__(self, *args, skill_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider
        cfg = self.config.algorithm.get("a7", {})
        self.a7_inner_lr = float(cfg.get("inner_lr", 1e-5))
        self.a7_teacher_precond = str(cfg.get("teacher_precond", "adamw"))
        self.a7_kappa = float(cfg.get("kappa", 0.3))
        self.a7_privileged = bool(cfg.get("privileged", True))
        self.a7_rho_brake = float(cfg.get("rho_brake", 0.3))
        self.a7_rho_boost = float(cfg.get("rho_boost", 0.1))
        self.a7_clip_c = float(cfg.get("clip_c", 3.0))
        self.a7_min_delta_std = float(cfg.get("min_delta_std", 1e-4))
        self.a7_skip_steps = int(cfg.get("skip_steps", 10))
        self.a7_warmdown_steps = int(cfg.get("warmdown_steps", 0))
        self.a7_teacher_micro_bsz = cfg.get("teacher_micro_bsz", None)
        if self.a7_teacher_micro_bsz is not None:
            self.a7_teacher_micro_bsz = int(self.a7_teacher_micro_bsz)
        if self.a7_privileged and self.skill_provider is None:
            raise ValueError("a7.privileged=True requires main_a7 to pass skill_provider")
        print(
            f"[A7] inner_lr={self.a7_inner_lr} kappa={self.a7_kappa} "
            f"precond={self.a7_teacher_precond} privileged={self.a7_privileged} | "
            f"rho_brake={self.a7_rho_brake} rho_boost={self.a7_rho_boost} "
            f"clip_c={self.a7_clip_c} skip={self.a7_skip_steps} warmdown={self.a7_warmdown_steps}"
        )

    def _a7_build_priv(self, batch):
        if not self.a7_privileged:
            return None
        return build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )

    def _a7_teacher_once(self, batch, fold, source_fold, priv):
        role = (fold != source_fold).to(torch.int64)
        teacher_input = batch.select(
            batch_keys=["responses", "input_ids", "attention_mask", "position_ids",
                        "old_log_probs", "advantages", "response_mask"]
        )
        teacher_input.batch["rlsfd_role"] = role
        if priv is not None:
            teacher_input.batch["teacher_input_ids"] = priv.batch["input_ids"]
            teacher_input.batch["teacher_attention_mask"] = priv.batch["attention_mask"]
            teacher_input.batch["teacher_position_ids"] = priv.batch["position_ids"]
        teacher_input.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if self.a7_teacher_micro_bsz is not None:
            teacher_input.meta_info["a7_teacher_micro_bsz"] = self.a7_teacher_micro_bsz
        teacher_input.meta_info["rlsfd_inner_lr"] = self.a7_inner_lr
        teacher_input.meta_info["rlsfd_teacher_precond"] = self.a7_teacher_precond
        teacher_input.meta_info["rlsfd_no_split"] = False
        teacher_input.meta_info["rlsfd_kappa"] = self.a7_kappa
        teacher_input.meta_info["rlsfd_target_role"] = 1
        out = self.actor_rollout_wg.compute_rlsfd_future_log_prob(teacher_input)
        inner = {f"a7/src{source_fold}/{k.split('/')[-1]}": float(v)
                 for k, v in out.meta_info.get("metrics", {}).items()}
        return out.batch["teacher_log_probs"], inner

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

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="A7 Training")
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

                    fold = make_crossfit_folds(
                        batch.non_tensor_batch["uid"], batch.non_tensor_batch["traj_uid"]
                    )
                    batch.batch["a7_fold"] = fold
                    metrics["a7/fold0_row_frac"] = float((fold == 0).float().mean().item())

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
                        fold = batch.batch["a7_fold"]
                        fold = fold.to(batch.batch["old_log_probs"].device)
                        n0 = int((fold == 0).sum().item())
                        n1 = int((fold == 1).sum().item())
                        slp = batch.batch["old_log_probs"]
                        tlp = torch.full_like(slp, float("nan"))
                        priv = self._a7_build_priv(batch) if (n0 > 0 and n1 > 0) else None
                        for src in (0, 1) if (n0 > 0 and n1 > 0) else ():
                            part, inner = self._a7_teacher_once(batch, fold, src, priv)
                            tgt = (fold == 1 - src).to(tlp.device)
                            part = part.to(device=tlp.device, dtype=tlp.dtype)
                            tlp[tgt] = part[tgt]
                            metrics.update(inner)
                        teacher_valid = torch.isfinite(tlp).all(dim=-1).to(slp.dtype)
                        tlp = torch.where(torch.isfinite(tlp), tlp, slp)
                        batch.batch["teacher_log_probs"] = tlp
                        batch.batch["a7_teacher_valid"] = teacher_valid
                        metrics["a7/fold0_rows"] = float(n0)
                        metrics["a7/fold1_rows"] = float(n1)

                    with _timer("a7_gate", timing_raw):
                        rho_brake_s = anneal_rho(self.global_steps, self.a7_rho_brake,
                                                 self.a7_skip_steps, self.a7_warmdown_steps)
                        rho_boost_s = anneal_rho(self.global_steps, self.a7_rho_boost,
                                                 self.a7_skip_steps, self.a7_warmdown_steps)
                        token_advantages, a7_metrics = compute_a7_twoside_advantage(
                            seq_advantages=batch.batch["advantages"],
                            student_log_probs=batch.batch["old_log_probs"],
                            teacher_log_probs=batch.batch["teacher_log_probs"],
                            response_mask=batch.batch["response_mask"],
                            teacher_valid=batch.batch["a7_teacher_valid"],
                            rho_brake_s=rho_brake_s,
                            rho_boost_s=rho_boost_s,
                            clip_c=self.a7_clip_c,
                            min_delta_std=self.a7_min_delta_std,
                        )
                        batch.batch["advantages"] = token_advantages
                        metrics.update(a7_metrics)

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
