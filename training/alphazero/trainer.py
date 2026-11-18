"""Gradient steps. train_policy_value: p02 pretrain + p03 iterations. distill_value: p04.

Data stays on the CPU as float16; each batch is mirrored (50%), moved to the device and
cast to float32 there (MPS has no float64; float32 everywhere is safe on cpu/cuda/mps)."""
import math

import torch
import torch.nn.functional as F

from .symmetry import FEATURE_MIRROR, POLICY_MIRROR

_FEAT = torch.tensor(FEATURE_MIRROR)
_POL = torch.tensor(POLICY_MIRROR)


def _maybe_mirror(xb, pib, enabled, generator):
    if not enabled:
        return xb, pib
    flip = (torch.rand(xb.shape[0], generator=generator) < 0.5)[:, None]
    xb = torch.where(flip, xb[:, _FEAT], xb)
    if pib is not None:
        pib = torch.where(flip, pib[:, _POL], pib)
    return xb, pib


def train_policy_value(net, data, cfg, device, writer, step, epochs, lr, tag):
    """Adam + cosine lr decay to cfg.lr_min. Loss = soft cross-entropy(policy) + MSE(value).
    Returns the next global step (TensorBoard x-axis continues across iterations)."""
    x, pi, z = data
    n = len(z)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, epochs * math.ceil(n / cfg.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=min(cfg.lr_min, lr))
    gen = torch.Generator().manual_seed(cfg.seed + step)
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=gen).split(cfg.batch_size):
            xb, pib = _maybe_mirror(x[idx], pi[idx], cfg.mirror_augment, gen)
            xb, pib, zb = xb.to(device).float(), pib.to(device).float(), z[idx].to(device).float()
            logits, value = net(xb)
            policy_loss = -(pib * F.log_softmax(logits, dim=1)).sum(1).mean()
            value_loss = F.mse_loss(value, zb)
            loss = policy_loss + value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf"))  # measure only
            opt.step()
            sched.step()
            if step % cfg.log_every == 0:
                with torch.no_grad():
                    decided = zb != 0
                    sign = ((value > 0) == (zb > 0))[decided].float().mean() if decided.any() else None
                    stats = {"policy_loss": policy_loss, "value_loss": value_loss, "loss": loss,
                             "policy_top1": (logits.argmax(1) == pib.argmax(1)).float().mean(),
                             "lr": sched.get_last_lr()[0], "grad_norm": grad_norm}
                    if sign is not None:
                        stats["value_sign_acc"] = sign
                for name, val in stats.items():
                    writer.add_scalar("%s/%s" % (tag, name), float(val.detach()) if torch.is_tensor(val) else float(val), step)
            step += 1
    net.eval()
    return step


@torch.no_grad()
def teacher_values(net, x, device, batch=4096):
    """Value-head outputs of an AZNet-like module for every row of x (CPU float32 tensor)."""
    net.to(device).eval()
    out = [net(x[i:i + batch].to(device).float())[1].float().cpu() for i in range(0, len(x), batch)]
    return torch.cat(out)


def pearson(a, b):
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def distill_value(tiny, x, targets, cfg, device, writer):
    """Fit tiny(x) to the teacher's values (MSE). Held-out split -> val MSE + Pearson per epoch."""
    gen = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(targets), generator=gen)
    n_val = max(1, int(len(targets) * cfg.distill_val_frac))
    val, train = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(tiny.parameters(), lr=cfg.lr)
    tiny.to(device)
    metrics = {}
    for epoch in range(cfg.distill_epochs):
        tiny.train()
        losses = []
        for idx in train[torch.randperm(len(train), generator=gen)].split(cfg.batch_size):
            xb, _ = _maybe_mirror(x[idx], None, cfg.mirror_augment, gen)   # value is mirror-invariant
            loss = F.mse_loss(tiny(xb.to(device).float()), targets[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        tiny.eval()
        with torch.no_grad():
            pred = tiny(x[val].to(device).float()).float().cpu()
        metrics = {"train_mse": sum(losses) / max(1, len(losses)),
                   "val_mse": float(F.mse_loss(pred, targets[val])),
                   "val_pearson": pearson(pred, targets[val])}
        for name, value in metrics.items():
            writer.add_scalar("distill/" + name, value, epoch)
    return metrics
