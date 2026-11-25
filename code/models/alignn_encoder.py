from __future__ import annotations

import torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from torch import nn


class ALIGNNEncoder(nn.Module):
    """ALIGNN 2026.4.2 backbone returning the pooled crystal embedding."""

    def __init__(
        self,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
        embedding_features: int = 64,
        hidden_features: int = 256,
    ) -> None:
        super().__init__()
        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=int(alignn_layers),
            gcn_layers=int(gcn_layers),
            embedding_features=int(embedding_features),
            hidden_features=int(hidden_features),
            output_features=int(hidden_features),
            classification=False,
            link="identity",
        )
        self.hidden_features = int(hidden_features)
        self.backbone = ALIGNN(config)
        # ALIGNN pools to h and then applies fc. Replacing fc by Identity
        # exposes exactly that pooled crystal representation.
        self.backbone.fc = nn.Identity()
        self.backbone.link = None

    def forward(self, graph_batch):
        embedding = self.backbone(graph_batch)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2 or embedding.shape[-1] != self.hidden_features:
            raise RuntimeError(
                "Unexpected ALIGNN embedding shape: "
                f"{tuple(embedding.shape)}; expected [batch, {self.hidden_features}]."
            )
        return embedding
