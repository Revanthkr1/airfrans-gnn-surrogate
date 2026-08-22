"""MeshGraphNets-style encode-process-decode GNN, in PyTorch Geometric."""
import torch
from torch import nn
from torch_geometric.utils import scatter


def mlp(in_dim, hidden_dim, out_dim, layernorm=True):
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    if layernorm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class GraphNetBlock(nn.Module):
    """One message-passing round: update edges from [src, dst, edge], then nodes from aggregated edges."""

    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = mlp(3 * latent_dim, hidden_dim, latent_dim)
        self.node_mlp = mlp(2 * latent_dim, hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        edge_out = edge_attr + self.edge_mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        agg = scatter(edge_out, dst, dim=0, dim_size=x.size(0), reduce="sum")
        node_out = x + self.node_mlp(torch.cat([x, agg], dim=-1))
        return node_out, edge_out


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        node_in_dim=7,
        edge_in_dim=2,
        out_dim=4,
        latent_dim=32,
        hidden_dim=64,
        n_message_passing=4,
    ):
        super().__init__()
        self.node_encoder = mlp(node_in_dim, hidden_dim, latent_dim)
        self.edge_encoder = mlp(edge_in_dim, hidden_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [GraphNetBlock(latent_dim, hidden_dim) for _ in range(n_message_passing)]
        )
        self.decoder = mlp(latent_dim, hidden_dim, out_dim, layernorm=False)

    def forward(self, node_features, edge_index, edge_attr):
        x = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for block in self.blocks:
            x, e = block(x, edge_index, e)
        return self.decoder(x)
