
import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
    positive_only: bool = False,
    opsd_uniform: bool = False,
    cut_negative: bool = False,
    empty_guard: bool = False,
) -> tuple[torch.Tensor, dict]:
    teacher_log_probs = teacher_log_probs.detach()

    delta_t = teacher_log_probs - student_log_probs.detach()

    pos_mask = (delta_t > 0).float()
    if opsd_uniform:
        gate = torch.ones_like(delta_t).detach()
    elif positive_only:
        gate = pos_mask.detach()
    else:
        gate = torch.sigmoid(gate_beta * delta_t).detach()
    if cut_negative and not positive_only:
        gate = (gate * pos_mask).detach()

    kl_per_token = teacher_log_probs - student_log_probs

    gated_kl = gate * kl_per_token

    _empty = False
    if empty_guard:
        _target_tokens = float((gate * response_mask).sum().item())
        _valid_tokens = float(response_mask.sum().item())
        if _target_tokens <= 0.0 or _valid_tokens <= 0.0:
            _empty = True

    if _empty:
        loss = torch.zeros((), dtype=gated_kl.dtype, device=gated_kl.device)
    else:
        loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * response_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        gap_mean = (delta_t * response_mask).sum() / mask_sum
        positive_frac = ((delta_t > 0).float() * response_mask).sum() / mask_sum

        _tt = (gate * response_mask).sum()
        empty_target_flag = 1.0 if bool(_empty) else 0.0
        target_token_frac = (_tt / mask_sum).item()

        neg_mask = (delta_t < 0).float()
        pm = pos_mask * response_mask
        nm = neg_mask * response_mask
        abs_d = delta_t.abs()
        pos_kl = (abs_d * pm).sum()
        neg_kl = (abs_d * nm).sum()
        tot_kl = (pos_kl + neg_kl).clamp(min=1e-8)
        pos_cnt = pm.sum().clamp(min=1)
        neg_cnt = nm.sum().clamp(min=1)
        tail = 1.0
        diag = {
            "sdar/pos_frac": (pm.sum() / mask_sum).item(),
            "sdar/neg_frac": (nm.sum() / mask_sum).item(),
            "sdar/pos_kl_share": (pos_kl / tot_kl).item(),
            "sdar/neg_kl_share": (neg_kl / tot_kl).item(),
            "sdar/pos_kl_mean": (pos_kl / pos_cnt).item(),
            "sdar/neg_kl_mean": (neg_kl / neg_cnt).item(),
            "sdar/pos_tail_ge1_frac": (((abs_d >= tail).float() * pm).sum() / pos_cnt).item(),
            "sdar/neg_tail_ge1_frac": (((abs_d >= tail).float() * nm).sum() / neg_cnt).item(),
            "sdar/pos_max_shock": (abs_d * pm).max().item(),
            "sdar/neg_max_shock": (abs_d * nm).max().item(),
        }

    metrics = {
        "sdar/gate_mean": gate_mean.item(),
        "sdar/gate_active_ratio": gate_active.item(),
        "sdar/positive_frac": positive_frac.item(),
        "sdar/empty_target": empty_target_flag,
        "sdar/target_token_frac": target_token_frac,
        "sdar/empty_guard_on": 1.0 if empty_guard else 0.0,
        "sdar/teacher_gap_mean": gap_mean.item(),
        "sdar/loss": loss.detach().item(),
        "sdar/cut_negative": float(cut_negative),
    }
    metrics.update(diag)

    return loss, metrics
