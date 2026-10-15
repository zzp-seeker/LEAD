
from typing import Dict, Tuple

import numpy as np
import torch


def make_crossfit_folds(uids: np.ndarray, traj_uids: np.ndarray) -> torch.Tensor:
    order: Dict[str, list] = {}
    for uid, tuid in zip(uids, traj_uids):
        lst = order.setdefault(str(uid), [])
        if tuid not in lst:
            lst.append(tuid)
    traj_fold: Dict[str, int] = {}
    for _uid, trajs in order.items():
        for k, tuid in enumerate(trajs):
            traj_fold[tuid] = k % 2
    folds = np.array([traj_fold[t] for t in traj_uids], dtype=np.int64)
    return torch.from_numpy(folds)


def compute_a7_twoside_advantage(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_valid: torch.Tensor,
    rho_brake_s: float,
    rho_boost_s: float,
    clip_c: float = 3.0,
    min_delta_std: float = 1e-4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        rm = response_mask.to(seq_advantages.dtype)
        tv = teacher_valid.to(seq_advantages.dtype).view(-1, 1)
        gate_mask = rm * tv

        delta = teacher_log_probs - student_log_probs
        n = gate_mask.sum().clamp(min=1.0)
        mu = (delta * gate_mask).sum() / n
        var = (((delta - mu) ** 2) * gate_mask).sum() / n
        std = var.clamp(min=0.0).sqrt()

        metrics = {
            "a7/delta_mean": float(mu),
            "a7/delta_std": float(std),
            "a7/rho_brake_s": float(rho_brake_s),
            "a7/rho_boost_s": float(rho_boost_s),
            "a7/teacher_valid_frac": float(tv.mean()),
        }

        if float(std) < min_delta_std or (rho_brake_s <= 0.0 and rho_boost_s <= 0.0):
            metrics.update({"a7/skipped": 1.0, "a7/w_mean": 1.0, "a7/adv_rewrite_frac": 0.0,
                            "a7/disagree_frac": 0.0, "a7/agree_boost_frac": 0.0,
                            "a7/agree_neg_hold_frac": 0.0})
            return seq_advantages * response_mask, metrics
        metrics["a7/skipped"] = 0.0

        dhat = ((delta - mu) / (std + 1e-8)).clamp(-clip_c, clip_c)

        first_valid = response_mask.int().argmax(dim=-1)
        bidx = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
        A_row = seq_advantages[bidx, first_valid].unsqueeze(-1)
        sgnA = torch.sign(A_row)
        metrics["a7/adv_row_nonconst_max"] = float(
            ((seq_advantages - A_row).abs() * rm).max()) if bool((rm > 0).any()) else 0.0

        sgn2 = dhat.sign() * sgnA
        boost = ((sgn2 > 0) & (A_row > 0)).to(seq_advantages.dtype) * gate_mask
        brake = (sgn2 < 0).to(seq_advantages.dtype) * gate_mask

        w = 1.0 + (rho_boost_s * boost - rho_brake_s * brake) * dhat.abs() / clip_c
        w = torch.where(gate_mask > 0, w, torch.ones_like(w))
        token_advantages = seq_advantages * w * response_mask

        nz = gate_mask > 0
        cnt = nz.sum().clamp(min=1)
        metrics.update({
            "a7/w_mean": float((w * gate_mask).sum() / cnt),
            "a7/w_min": float(w[nz].min()) if bool(nz.any()) else 1.0,
            "a7/w_max": float(w[nz].max()) if bool(nz.any()) else 1.0,
            "a7/disagree_frac": float(brake.sum() / cnt),
            "a7/agree_boost_frac": float(boost.sum() / cnt),
            "a7/agree_neg_hold_frac": float((((sgn2 > 0) & (A_row < 0)).to(w.dtype) * gate_mask).sum() / cnt),
            "a7/adv_rewrite_frac": float(((w - 1.0).abs() > 1e-8).to(w.dtype).sum() / cnt),
            "a7/dhat_abs_mean": float((dhat.abs() * gate_mask).sum() / cnt),
            "a7/adv_zero_frac": float(((sgnA == 0).to(w.dtype) * gate_mask).sum() / cnt),
        })
        return token_advantages, metrics


def anneal_rho(step: int, rho0: float, skip_steps: int, warmdown_steps: int) -> float:
    if step <= skip_steps:
        return 0.0
    if warmdown_steps and warmdown_steps > 0:
        return max(rho0 * (1.0 - (step - skip_steps) / float(warmdown_steps)), 0.0)
    return rho0
