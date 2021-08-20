"""Implementation of various metrics related to model saturation."""

import torch

from .saturate import saturate


@torch.no_grad()
def get_norm(model):
    # Use the same norm as for T5.
    params = [p for p in model.parameters() if p.requires_grad]
    params = [p for p in params if len(p.shape) > 0]
    return torch.cat([p.flatten() for p in params]).norm(p=2)


@torch.no_grad()
def get_saturation(soft, model, hard_callback):
    with saturate(model):
        hard = hard_callback()
    prod = torch.einsum("bti, bti -> bt", soft, hard)
    soft_norm = soft.norm(p=2, dim=-1)
    hard_norm = hard.norm(p=2, dim=-1)
    return prod / (soft_norm * hard_norm + 1e-9)