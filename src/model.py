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
        # Normalizes the SCALE of x reaching the decoder, not its output --
        # the decoder's own weights/biases can still freely map a normalized
        # input to an unconstrained output range, so this doesn't fight the
        # decoder's own lack of LayerNorm (section 6: forcing the *output*
        # to zero-mean-unit-variance would fight learning the real target
        # distribution -- normalizing the *input* doesn't do that). Added
        # after repeated detect_anomaly=True runs on the radius-subsampling
        # regime all pinpointed the same NaN at this exact op
        # (self.decoder's first Addmm) despite fixing three separate,
        # verified, unrelated data bugs (edge_attr scale, float16 target
        # overflow, cross-body radius-graph edges) -- none of which fully
        # resolved it. Each residual add across n_message_passing rounds is
        # itself LayerNorm-bounded, but their SUM into x is not, and a local
        # probe (ARCHITECTURE.md section 11) showed x's scale genuinely
        # drifting upward over real training steps rather than staying
        # fixed -- consistent with "eventually some batch pushes it past
        # the decoder's unprotected Linear layers," which no amount of
        # upstream data hygiene alone can rule out.
        self.pre_decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder = mlp(latent_dim, hidden_dim, out_dim, layernorm=False)

    def forward(self, node_features, edge_index, edge_attr):
        x = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for block in self.blocks:
            x, e = block(x, edge_index, e)
        return self.decoder(self.pre_decoder_norm(x))
