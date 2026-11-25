from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


TASK_INDEX = {"thermo": 0, "dynamic": 1, "joint": 2}


class MultiTaskObjective(nn.Module):
    """Joint-primary multi-task objective.

    Learnable ``log_vars`` are global task-level loss-balancing parameters.
    They must not be interpreted as per-material predictive uncertainty.
    """

    def __init__(
        self,
        active_tasks: list[str],
        weighting: str,
        positive_weights: dict[str, float],
        fixed_weights: dict[str, float] | None = None,
        logic_beta: float = 0.0,
        logic_lower_bound: bool = True,
    ) -> None:
        super().__init__()
        self.active_tasks = list(active_tasks)
        self.weighting = str(weighting)
        self.fixed_weights = fixed_weights or {}
        self.logic_beta = float(logic_beta)
        self.logic_lower_bound = bool(logic_lower_bound)

        for task in TASK_INDEX:
            self.register_buffer(
                f"positive_weight_{task}",
                torch.tensor(float(positive_weights.get(task, 1.0))),
            )

        if self.weighting == "adaptive_auxiliary":
            adaptive = [task for task in self.active_tasks if task != "joint"]
        elif self.weighting == "adaptive_all":
            adaptive = list(self.active_tasks)
        else:
            adaptive = []
        self.log_vars = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(())) for task in adaptive}
        )

    def binary_loss(
        self,
        task: str,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        positive_weight = getattr(self, f"positive_weight_{task}")
        return F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=positive_weight,
        )

    def forward(
        self,
        logits: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw: dict[str, torch.Tensor] = {}
        for task in self.active_tasks:
            raw[task] = self.binary_loss(
                task,
                logits[task],
                labels[:, TASK_INDEX[task]].float(),
            )

        total = torch.zeros((), device=labels.device)
        if self.weighting == "equal":
            for loss in raw.values():
                total = total + loss
        elif self.weighting == "fixed":
            for task, loss in raw.items():
                total = total + float(self.fixed_weights.get(task, 1.0)) * loss
        elif self.weighting in {"adaptive_auxiliary", "adaptive_all"}:
            for task, loss in raw.items():
                if task in self.log_vars:
                    log_var = self.log_vars[task]
                    total = total + torch.exp(-log_var) * loss + log_var
                else:
                    total = total + loss
        else:
            raise ValueError(f"Unknown task-weighting strategy: {self.weighting}")

        logic = torch.zeros((), device=labels.device)
        required = {"thermo", "dynamic", "joint"}
        if self.logic_beta > 0 and required.issubset(logits):
            thermo = torch.sigmoid(logits["thermo"])
            dynamic = torch.sigmoid(logits["dynamic"])
            joint = torch.sigmoid(logits["joint"])
            # Intersection probability cannot exceed either marginal.
            upper_violation = (
                F.relu(joint - thermo).mean()
                + F.relu(joint - dynamic).mean()
            )
            logic = upper_violation
            if self.logic_lower_bound:
                # Fréchet lower bound: P(A∩B) >= max(0, P(A)+P(B)-1).
                logic = logic + F.relu(thermo + dynamic - 1.0 - joint).mean()
            total = total + self.logic_beta * logic

        diagnostics: dict[str, torch.Tensor] = {
            f"loss_{task}": value.detach() for task, value in raw.items()
        }
        diagnostics["loss_logic"] = logic.detach()
        for task, value in self.log_vars.items():
            diagnostics[f"log_var_{task}"] = value.detach()
            diagnostics[f"effective_weight_{task}"] = torch.exp(-value.detach())
        return total, diagnostics
