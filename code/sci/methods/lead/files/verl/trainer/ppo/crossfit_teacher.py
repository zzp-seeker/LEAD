
from collections import defaultdict

import torch


def make_crossfit_folds(uids):
    seen = defaultdict(int)
    counts = defaultdict(int)
    folds = []
    for uid in uids:
        key = str(uid)
        folds.append(seen[key] % 2)
        seen[key] += 1
        counts[key] += 1
    bad = {uid: count for uid, count in counts.items() if count < 2 or count % 2 != 0}
    if bad:
        raise ValueError(f"cross-fit requires an even rollout count >=2 for every uid; got {bad}")
    return folds


@torch.no_grad()
def apply_crossfit_shadow_step(actor_module, actor_optimizer, inner_lr, kappa):
    if not 0.0 <= float(kappa) <= 1.0:
        raise ValueError(f"sfdamp_batch_weight must be in [0,1], got {kappa}")

    p2sg = {}
    for group in actor_optimizer.param_groups:
        for param in group["params"]:
            p2sg[param] = (actor_optimizer.state.get(param, {}), group)

    saved = []
    delta_sq = dot = grad_sq = hist_sq = 0.0
    for param in actor_module.parameters():
        if param.grad is None or param not in p2sg:
            continue
        state, group = p2sg[param]
        grad = param.grad
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-8)
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None or exp_avg_sq is None:
            delta = grad.mul(-inner_lr)
        else:
            step_t = state.get("step")
            step = float(step_t.item()) if torch.is_tensor(step_t) else float(step_t or 0)
            step += 1.0

            hist_hat = exp_avg.div(1.0 - beta1**step)
            dot += float((grad.float() * hist_hat.float()).sum().item())
            grad_sq += float(grad.float().pow(2).sum().item())
            hist_sq += float(hist_hat.float().pow(2).sum().item())

            mixed = exp_avg.mul(1.0 - kappa).add_(grad, alpha=kappa)
            second = exp_avg_sq.mul(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias1 = 1.0 - (1.0 - kappa) ** step
            if bias1 < 1e-12:
                bias1 = 1.0 - beta1**step
            mixed_hat = mixed.div_(bias1)
            denom = second.div_(1.0 - beta2**step).sqrt_().add_(eps)
            delta = mixed_hat.div_(denom).mul_(-inner_lr)

        saved.append((param, param.detach().clone()))
        param.add_(delta)
        delta_sq += float(delta.float().pow(2).sum().item())

    cosine = dot / ((grad_sq**0.5) * (hist_sq**0.5) + 1e-12) if grad_sq and hist_sq else 0.0
    return saved, delta_sq**0.5, cosine

