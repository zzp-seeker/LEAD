import torch

from verl.trainer.ppo.core_algos import agg_loss


def rlsd_effective_lam(lam0, global_steps, warmdown_steps):
    lam0 = float(lam0)
    warmdown_steps = int(warmdown_steps or 0)
    if warmdown_steps <= 0 or global_steps is None:
        return lam0
    step_index = max(int(global_steps) - 1, 0)
    return max(lam0 * (1.0 - min(step_index / warmdown_steps, 1.0)), 0.0)


def compute_rlsd_loss(
    log_prob, old_log_prob, teacher_log_prob, advantages, response_mask,
    clip_ratio=0.2, epsilon_w=0.2, lam=0.5, valid_mask=None,
    loss_agg_mode="token-mean", rollout_is_weights=None,
    shock_cfg=None, student_entropy=None,
    cut_negative=False,
):
    with torch.no_grad():
        delta_t = teacher_log_prob - log_prob.detach()
        _w_lower = 1.0 if cut_negative else (1.0 - epsilon_w)
        w_t = (torch.sign(advantages) * delta_t).exp().clamp(_w_lower, 1.0 + epsilon_w)
        if valid_mask is not None:
            w_t = torch.where(valid_mask > 0, w_t, torch.ones_like(w_t))
        modified_adv = advantages * ((1.0 - lam) + lam * w_t)

    log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    surr1 = ratio * modified_adv
    surr2 = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * modified_adv
    per_token = -torch.min(surr1, surr2)
    if rollout_is_weights is not None:
        per_token = per_token * rollout_is_weights
    loss = agg_loss(loss_mat=per_token, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    if response_mask.sum() == 0:
        loss = per_token.sum() * 0.0

    _shock = {}
    if shock_cfg is not None:
        try:
            from verl.trainer.ppo.core_algos import exp3_shock_metrics
            _vm = valid_mask * response_mask if valid_mask is not None else response_mask
            exp3_shock_metrics(_shock, teacher_log_prob, log_prob.detach(), _vm,
                               shock_cfg, student_entropy)
        except Exception as _e:
            print(f"[exp3/shock@rlsd] skipped: {_e!r}")

    with torch.no_grad():
        ms = response_mask.sum().clamp(min=1)
        vm = valid_mask * response_mask if valid_mask is not None else response_mask
        n_valid = vm.sum()
        vs = n_valid.clamp(min=1)
        if n_valid == 0:
            metrics = {
                "rlsd/teacher_gap_mean": 0.0, "rlsd/teacher_gap_abs_mean": 0.0,
                "rlsd/w_mean": 1.0, "rlsd/w_dev_mean": 0.0, "rlsd/w_clipfrac": 0.0,
                "rlsd/valid_frac": 0.0, "rlsd/lam": float(lam),
                "rlsd/pg_clipfrac": ((((ratio > 1.0 + clip_ratio) | (ratio < 1.0 - clip_ratio)).float() * response_mask).sum() / ms).item(),
            }
            metrics.update(_shock)
            return loss, metrics
        metrics = {
            "rlsd/teacher_gap_mean": ((delta_t * vm).sum() / vs).item(),
            "rlsd/teacher_gap_abs_mean": ((delta_t.abs() * vm).sum() / vs).item(),
            "rlsd/w_mean": ((w_t * vm).sum() / vs).item(),
            "rlsd/w_dev_mean": (((w_t - 1.0).abs() * vm).sum() / vs).item(),
            "rlsd/w_clipfrac": ((((w_t <= 1.0 - epsilon_w + 1e-9) | (w_t >= 1.0 + epsilon_w - 1e-9)).float() * vm).sum() / vs).item(),
            "rlsd/valid_frac": (vm.sum() / ms).item(),
            "rlsd/lam": float(lam),
            "rlsd/cut_negative": float(cut_negative),
            "rlsd/cutneg_lifted_frac": ((((torch.sign(advantages) * delta_t).exp() < 1.0).float() * vm).sum() / vs).item() if cut_negative else 0.0,
            "rlsd/pg_clipfrac": ((((ratio > 1.0 + clip_ratio) | (ratio < 1.0 - clip_ratio)).float() * response_mask).sum() / ms).item(),
        }
    metrics.update(_shock)
    return loss, metrics
