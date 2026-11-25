"""Native joint distributions used by the paper revision controls."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def four_state_logits(state_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """State order is (T,D) = 00,01,10,11; compute binary log odds stably."""
    if state_logits.ndim != 2 or state_logits.shape[1] != 4:
        raise ValueError("Expected four-state logits with shape [batch, 4].")
    return {
        "thermo": torch.logsumexp(state_logits[:, [2, 3]], 1)
        - torch.logsumexp(state_logits[:, [0, 1]], 1),
        "dynamic": torch.logsumexp(state_logits[:, [1, 3]], 1)
        - torch.logsumexp(state_logits[:, [0, 2]], 1),
        "joint": state_logits[:, 3] - torch.logsumexp(state_logits[:, :3], 1),
        "state_logits": state_logits,
    }


def product_joint_logit(thermo: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
    # pT*pD / (1-pT*pD) = exp(T+D)/(1+exp(T)+exp(D)).
    return thermo + dynamic - torch.logsumexp(
        torch.stack([torch.zeros_like(thermo), thermo, dynamic]), dim=0
    )


class FourStateObjective(torch.nn.Module):
    def forward(self, logits: dict, labels: torch.Tensor):
        if labels.ndim != 2 or labels.shape[1] != 3:
            raise ValueError("Expected binary T,D,J labels with shape [batch, 3].")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("Four-state CE requires complete binary labels.")
        if not torch.equal(labels[:, 2], labels[:, 0] * labels[:, 1]):
            raise ValueError("Inconsistent joint labels: J must equal T AND D.")
        target = (2 * labels[:, 0] + labels[:, 1]).long()
        loss = F.cross_entropy(logits["state_logits"], target)
        return loss, {"loss_four_state_ce": loss.detach()}
