from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from dgl.nn import AvgPooling
from torch import nn
from torch.nn import functional as F


class GaussianDistance(nn.Module):
    def __init__(self, dmin: float, dmax: float, bins: int) -> None:
        super().__init__()
        centers = torch.linspace(float(dmin), float(dmax), int(bins))
        self.register_buffer("centers", centers)
        step = (float(dmax) - float(dmin)) / max(int(bins) - 1, 1)
        self.width = max(step, 1e-6)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -((distances.unsqueeze(-1) - self.centers) ** 2) / (self.width**2)
        )


class CGCNNConv(nn.Module):
    """CGCNN-style gated crystal graph convolution.

    This follows the original CGCNN gating structure while using DGL graphs
    already generated for ALIGNN, ensuring identical neighborhoods across
    graph-model baselines.
    """

    def __init__(self, hidden_features: int, bond_features: int) -> None:
        super().__init__()
        total = 2 * hidden_features + bond_features
        self.message = nn.Linear(total, 2 * hidden_features)
        self.message_bn = nn.BatchNorm1d(2 * hidden_features)
        self.aggregate_bn = nn.BatchNorm1d(hidden_features)

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: torch.Tensor,
        bond_features: torch.Tensor,
    ) -> torch.Tensor:
        graph = graph.local_var()
        graph.ndata["node"] = node_features
        graph.edata["bond"] = bond_features

        def edge_message(edges):
            z = torch.cat(
                [edges.src["node"], edges.dst["node"], edges.data["bond"]],
                dim=-1,
            )
            transformed = self.message_bn(self.message(z))
            gate, core = transformed.chunk(2, dim=-1)
            return {"message": torch.sigmoid(gate) * F.softplus(core)}

        graph.apply_edges(edge_message)
        graph.update_all(fn.copy_e("message", "m"), fn.sum("m", "aggregate"))
        aggregate = self.aggregate_bn(graph.ndata["aggregate"])
        return F.softplus(node_features + aggregate)


class CGCNNEncoder(nn.Module):
    def __init__(
        self,
        atom_input_features: int = 92,
        hidden_features: int = 256,
        layers: int = 3,
        bond_bins: int = 41,
        bond_max: float = 8.0,
    ) -> None:
        super().__init__()
        self.hidden_features = int(hidden_features)
        self.atom_projection = nn.Linear(atom_input_features, hidden_features)
        self.distance_expansion = GaussianDistance(0.0, bond_max, bond_bins)
        self.convolutions = nn.ModuleList(
            [
                CGCNNConv(hidden_features, bond_bins)
                for _ in range(int(layers))
            ]
        )
        self.readout = AvgPooling()

    def forward(self, graph_batch):
        graph, _line_graph, _lattice = graph_batch
        graph = graph.local_var()
        node = F.softplus(self.atom_projection(graph.ndata["atom_features"]))
        distance = torch.linalg.norm(graph.edata["r"], dim=1)
        bond = self.distance_expansion(distance)
        for convolution in self.convolutions:
            node = convolution(graph, node, bond)
        return self.readout(graph, node)
