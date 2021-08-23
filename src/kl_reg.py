from typing import Optional
import torch
from torch.nn import KLDivLoss


class KlSatReg:
    """A regularization method that heuristically incentivizes a probability distribution towards a saturated one.
    
    By default, uses the KL divergence between probs and heuristically saturated probs as a loss term. The
    heuristic saturation is designed to tolerate imprecision, i.e., treat values that are close as the same.
    """

    def __init__(self, loss: Optional["torch._Loss"] = None, tol: float = .9):
        self.loss = loss or KLDivLoss(reduction="sum")  # The loss should return a sum, not a mean.
        self.tol = tol

    def __call__(self, probs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # assert not mask
        # mask = mask.unsqueeze(dim=1)
        # mask = mask.unsqueeze(dim=2) * mask.unsqueeze(dim=3)
        with torch.no_grad():
            max_prob, _ = (probs * mask).max(dim=-1)
            max_prob = max_prob.unsqueeze(dim=-1)
            sat_mask = (probs > max_prob * self.tol)
            counts = sat_mask.sum(dim=-1).unsqueeze(dim=-1)
            sat_probs = sat_mask.float() / counts
        # Add dimension for heads and for probabilities.
        breakpoint()
        return self.loss(probs, sat_probs) / mask.sum()
