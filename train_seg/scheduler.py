# ============================================================
# Learning Rate Scheduler (Cosine with Warmup)
# ============================================================

import math
import torch


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Cosine annealing LR scheduler with linear warmup."""
    total = cfg.train.epochs * steps_per_epoch
    warmup = cfg.train.warmup_epochs * steps_per_epoch

    def lr_fn(step):
        if step < warmup:
            return max(1e-6, step / warmup)
        t = (step - warmup) / max(total - warmup, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
