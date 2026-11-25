from __future__ import annotations

import torch
from torch import nn

from models.alignn_encoder import ALIGNNEncoder
from models.cgcnn import CGCNNEncoder
from models.joint_outputs import four_state_logits, product_joint_logit


VALID_TASKS = ("thermo", "dynamic", "joint")


class CrystalMultiTaskModel(nn.Module):
    def __init__(
        self,
        model_cfg: dict,
        heads: list[str],
        model_type: str = "alignn",
    ) -> None:
        super().__init__()
        invalid = set(heads) - set(VALID_TASKS)
        if invalid:
            raise ValueError(f"Invalid task heads: {sorted(invalid)}")
        if not heads:
            raise ValueError("At least one prediction head is required.")
        self.head_names = list(heads)
        self.output_mode = model_cfg.get("output_mode", "independent")
        if self.output_mode not in {"independent", "four_state", "dual_product"}:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
        if self.output_mode == "dual_product" and heads != ["thermo", "dynamic"]:
            raise ValueError("dual_product requires exactly thermo and dynamic heads.")
        if self.output_mode == "four_state" and heads != list(VALID_TASKS):
            raise ValueError("four_state exposes exactly thermo, dynamic, joint.")
        self.prediction_tasks = (
            list(VALID_TASKS) if self.output_mode != "independent" else list(heads)
        )
        self.conditioning_features = list(
            model_cfg.get("conditioning_features", [])
        )
        hidden = int(model_cfg["hidden_features"])
        if model_type == "alignn":
            self.encoder = ALIGNNEncoder(
                alignn_layers=int(model_cfg["alignn_layers"]),
                gcn_layers=int(model_cfg["gcn_layers"]),
                embedding_features=int(model_cfg["embedding_features"]),
                hidden_features=hidden,
            )
        elif model_type == "cgcnn":
            self.encoder = CGCNNEncoder(
                atom_input_features=92,
                hidden_features=hidden,
                layers=int(model_cfg["cgcnn_layers"]),
                bond_bins=int(model_cfg["cgcnn_bond_bins"]),
                bond_max=float(model_cfg["cgcnn_bond_max"]),
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        dropout = float(model_cfg.get("dropout", 0.0))
        self.embedding_dropout = nn.Dropout(dropout)
        head_input_features = hidden + len(self.conditioning_features)
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(head_input_features, hidden // 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 1),
                )
                for task in (self.head_names if self.output_mode != "four_state" else [])
            }
        )
        if self.output_mode == "four_state":
            self.state_head = nn.Sequential(
                nn.Linear(head_input_features, hidden // 2), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(hidden // 2, 4),
            )

    def forward(self, graph_batch, return_embedding: bool = False):
        if self.conditioning_features:
            if len(graph_batch) != 4:
                raise ValueError(
                    "A conditioning tensor is required for model features "
                    f"{self.conditioning_features}."
                )
            encoder_batch = graph_batch[:3]
            conditioning = graph_batch[3]
        else:
            encoder_batch = graph_batch[:3]
            conditioning = None

        embedding = self.embedding_dropout(self.encoder(encoder_batch))
        head_embedding = embedding
        if conditioning is not None:
            if conditioning.ndim == 1:
                conditioning = conditioning.unsqueeze(-1)
            if conditioning.shape[0] != embedding.shape[0]:
                raise ValueError(
                    "Conditioning batch size does not match crystal embeddings."
                )
            if conditioning.shape[1] != len(self.conditioning_features):
                raise ValueError(
                    "Conditioning feature width does not match model configuration."
                )
            head_embedding = torch.cat([embedding, conditioning], dim=-1)
        output_mode = getattr(self, "output_mode", "independent")
        if output_mode == "four_state":
            logits = four_state_logits(self.state_head(head_embedding))
        else:
            logits = {
                task: self.heads[task](head_embedding).squeeze(-1)
                for task in self.head_names
            }
            if output_mode == "dual_product":
                logits["joint"] = product_joint_logit(logits["thermo"], logits["dynamic"])
        if return_embedding:
            return logits, embedding
        return logits
