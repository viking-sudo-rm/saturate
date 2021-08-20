from typing import Optional
import torch
from torch.nn import KLDivLoss


class KlSatReg:
    """A regularization method that heuristically incentivizes a probability distribution towards a saturated one.
    
    By default, uses the KL divergence between probs and heuristically saturated probs as a loss term. The
    heuristic saturation is designed to tolerate imprecision, i.e., treat values that are close as the same.
    """

    def __init__(self, loss: Optional["torch._Loss"] = None, tol: float = .9, detach: bool = True):
        self.loss = loss or KLDivLoss()
        self.tol = tol
        self.detach = detach

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        max_prob, _ = probs.max(dim=-1)
        max_prob = max_prob.unsqueeze(dim=-1)
        if self.detach:
            max_prob = max_prob.detach()
        mask = (probs > max_prob * self.tol)
        counts = mask.sum(dim=-1).unsqueeze(dim=-1)
        sat_probs = mask.float() / counts
        return self.loss(probs, sat_probs)
