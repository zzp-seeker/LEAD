
from typing import Dict, Tuple

import numpy as np
import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1)


def compute_rlsfd_role(uids: np.ndarray, traj_uids: np.ndarray) -> torch.Tensor:
    group_trajs: Dict[str, list] = {}
    for uid, tuid in zip(uids, traj_uids):
        lst = group_trajs.setdefault(uid, [])
        if tuid not in lst:
            lst.append(tuid)
    traj_role: Dict[str, int] = {}
    for uid, trajs in group_trajs.items():
        half = len(trajs) // 2
        for k, tuid in enumerate(trajs):
            traj_role[tuid] = 0 if k < half else 1
    roles = np.array([traj_role[t] for t in traj_uids], dtype=np.int64)
    return torch.from_numpy(roles)


def compute_rlsfd_modulated_advantage(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    role: torch.Tensor,
    w_coef: float = 0.2,
    epsilon_w: float = 0.2,
    mod_center: str = "seq",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        role_t = role.to(dtype=seq_advantages.dtype).view(-1, 1)
        d_mask = response_mask * role_t

        first_valid = response_mask.int().argmax(dim=-1)
        bidx = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
        A_row = seq_advantages[bidx, first_valid]
        sign_A = torch.sign(A_row).unsqueeze(-1)

        gap = teacher_log_probs - student_log_probs

        gap_scale = masked_mean(gap.abs(), d_mask)
        ghat = gap / (gap_scale + 1e-8)
        if mod_center == "seq":
            cnt = d_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            ghat = ghat - (ghat * d_mask).sum(dim=-1, keepdim=True) / cnt

        w_exp = (sign_A * w_coef * ghat).clamp(min=-20.0, max=20.0)
        w_raw = w_exp.exp()
        w = w_raw.clamp(1.0 - epsilon_w, 1.0 + epsilon_w)
        if mod_center == "seq":
            w_row_mean = ((w * d_mask).sum(dim=-1, keepdim=True) / cnt).clamp(min=1e-6)
            w = torch.where(d_mask > 0, w / w_row_mean, w)

        token_advantages = seq_advantages * ((1.0 - role_t) + role_t * w)
        token_advantages = token_advantages * response_mask

        rl_mask = response_mask * (1.0 - role_t)
        metrics = {
            "rlsfd/gap_mean": masked_mean(gap, d_mask).item(),
            "rlsfd/gap_abs_mean": masked_mean(gap.abs(), d_mask).item(),
            "rlsfd/gap_pos_frac": masked_mean((gap > 0).to(gap.dtype), d_mask).item(),
            "rlsfd/w_clip_frac": masked_mean(
                ((w_raw < 1.0 - epsilon_w) | (w_raw > 1.0 + epsilon_w)).to(gap.dtype), d_mask
            ).item(),
            "rlsfd/w_dev_mean": masked_mean((w - 1.0).abs(), d_mask).item(),
            "rlsfd/w_row_mean_dev": masked_mean(
                (((w * d_mask).sum(-1) / d_mask.sum(-1).clamp(min=1)) - 1.0).abs().unsqueeze(-1)
                * (d_mask.sum(-1, keepdim=True) > 0),
                (d_mask.sum(-1, keepdim=True) > 0).to(gap.dtype),
            ).item(),
            "rlsfd/distill_token_frac": (d_mask.sum() / response_mask.sum().clamp(min=1)).item(),
            "rlsfd/rl_adv_abs_mean": masked_mean(seq_advantages.abs(), rl_mask).item(),
            "rlsfd/mod_adv_abs_mean": masked_mean(
                (seq_advantages * role_t * (w - 1.0)).abs(), d_mask
            ).item(),
        }
    return token_advantages, metrics
