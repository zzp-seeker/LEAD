import torch


@torch.no_grad()
def apply_momentum_shadow_step(actor_module, actor_optimizer, inner_lr):
    p2sg = {}
    for group in actor_optimizer.param_groups:
        for p in group["params"]:
            p2sg[p] = (actor_optimizer.state.get(p, {}), group)
    saved, delta_sq = [], 0.0
    for p in actor_module.parameters():
        sg = p2sg.get(p)
        if sg is None:
            continue
        state, group = sg
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None or exp_avg_sq is None:
            continue
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-8)
        step_t = state.get("step")
        step = float(step_t.item()) if torch.is_tensor(step_t) else float(step_t or 0)
        if step < 1:
            continue
        m_hat = exp_avg.div(1.0 - beta1 ** step)
        denom = exp_avg_sq.div(1.0 - beta2 ** step).sqrt_().add_(eps)
        delta = m_hat.div_(denom).mul_(-inner_lr)
        saved.append((p, p.detach().clone()))
        p.add_(delta)
        delta_sq += float(delta.float().pow(2).sum().item())
    return saved, delta_sq ** 0.5


@torch.no_grad()
def _readonly_adamw_delta(param, state, group, inner_lr):
    if param.grad is None:
        return None
    grad = param.grad
    beta1, beta2 = group.get("betas", (0.9, 0.999))
    eps = group.get("eps", 1e-8)
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg is None or exp_avg_sq is None:
        return grad.mul(-inner_lr)
    step_t = state.get("step")
    step = float(step_t.item()) if torch.is_tensor(step_t) else float(step_t or 0)
    step += 1.0
    m = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1)
    v = exp_avg_sq.mul(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    m_hat = m.div_(1.0 - beta1 ** step)
    denom = v.div_(1.0 - beta2 ** step).sqrt_().add_(eps)
    return m_hat.div_(denom).mul_(-inner_lr)


@torch.no_grad()
def apply_grad_shadow_step(actor_module, actor_optimizer, inner_lr):
    p2sg = {}
    for group in actor_optimizer.param_groups:
        for p in group["params"]:
            p2sg[p] = (actor_optimizer.state.get(p, {}), group)
    saved, delta_sq = [], 0.0
    for p in actor_module.parameters():
        if p.grad is None or p not in p2sg:
            continue
        state, group = p2sg[p]
        delta = _readonly_adamw_delta(p, state, group, inner_lr)
        if delta is None:
            continue
        saved.append((p, p.detach().clone()))
        p.add_(delta)
        delta_sq += float(delta.float().pow(2).sum().item())
    return saved, delta_sq ** 0.5


def restore_params(saved):
    with torch.no_grad():
        for p, orig in saved:
            p.copy_(orig)
    saved.clear()
