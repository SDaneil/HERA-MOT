"""Four-frame connection formation with fixed-depth predicted-partition feedback."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..tracker.graph import conditional_paths, decode_graph


@dataclass(frozen=True)
class RefinementConfig:
    appearance_dim: int = 2048
    dim: int = 64
    heads: int = 4
    rounds: int = 3


@dataclass
class ConnectionRound:
    gains: torch.Tensor
    partition: object
    path_indices: np.ndarray  # Context used to produce this round, not its output.


class ConnectionRefinement(nn.Module):
    def __init__(self, config=RefinementConfig()):
        super().__init__()
        if config.rounds < 1 or config.dim % config.heads:
            raise ValueError('Invalid refinement configuration')
        self.config = config
        d = config.dim
        self.detector = nn.Sequential(nn.Linear(config.appearance_dim+5, d), nn.LayerNorm(d), nn.ReLU())
        self.time = nn.Embedding(4, d)
        self.context = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, 2*d, dropout=0.), 1, enable_nested_tensor=False)
        self.edge_input = nn.Sequential(nn.Linear(2*d+7, d), nn.LayerNorm(d), nn.ReLU())
        self.motion = nn.Linear(6, d)
        self.path_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, 2*d, dropout=0.), 1, enable_nested_tensor=False)
        self.node_update = nn.GRUCell(2*d, d)
        self.edge_update = nn.GRUCell(3*d+1, d)
        self.edge_score = nn.Linear(d, 1)
        self.no_in = nn.Linear(d, 1)
        self.no_out = nn.Linear(d, 1)

    def _gains(self, nodes, edges, endpoints):
        u, v = endpoints.T
        return (self.edge_score(edges).squeeze(-1)
                - self.no_out(nodes).squeeze(-1)[u]
                - self.no_in(nodes).squeeze(-1)[v])

    def _competition(self, nodes, edges, endpoints, gains):
        n, d = nodes.shape
        u, v = endpoints.T
        dense = gains.new_full((n, n), -float('inf'))
        dense[u, v] = gains
        null = gains.new_zeros((n, 1))
        out_weights = torch.cat((dense, null), 1).softmax(1)[u, v]
        in_weights = torch.cat((dense.T, null), 1).softmax(1)[v, u]
        outgoing = nodes.new_zeros((n, d)).index_add(0, u, out_weights[:, None]*edges)
        incoming = nodes.new_zeros((n, d)).index_add(0, v, in_weights[:, None]*edges)
        return self.node_update(torch.cat((incoming, outgoing), -1), nodes)

    def _paths(self, window, nodes, indices):
        device = nodes.device
        ix = torch.as_tensor(indices, device=device)
        mask = ix < 0
        safe = ix.clamp_min(0)
        boxes = torch.as_tensor(window.bboxes, dtype=nodes.dtype, device=device)[safe]
        frames = torch.as_tensor(window.frames, dtype=nodes.dtype, device=device)[safe]
        dt = (frames[:, 1:] - frames[:, :-1]).clamp_min(1)
        centers = boxes[..., :2] + boxes[..., 2:]/2
        height = ((boxes[:, 1:, 3] + boxes[:, :-1, 3])/2).clamp_min(1)
        velocity = (centers[:, 1:] - centers[:, :-1]) / (height*dt)[..., None]
        size_rate = (boxes[:, 1:, 2:] / boxes[:, :-1, 2:]).log() / dt[..., None]
        motion = nodes.new_zeros((*indices.shape, 6))
        motion[:, 1:] = torch.cat((velocity, size_rate, (dt/3)[..., None],
                                   (dt > 1).to(nodes.dtype)[..., None]), -1)
        tokens = nodes[safe] + self.motion(motion)
        encoded = self.path_encoder(tokens.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)
        return encoded.masked_fill(mask[..., None], 0.).sum(1) / (~mask).sum(1)[:, None]

    def forward(self, window, graph, feedback=True, shared_context=True):
        if window.window_size > 4 or not np.array_equal(graph.frames, window.frames):
            raise ValueError('Graph/window mismatch or window wider than four')
        device = next(self.parameters()).device
        if not graph.size:
            # Keep a differentiable zero for empty-window training batches.
            gains = self.edge_score.weight.sum().expand(0)
            return [ConnectionRound(gains, decode_graph(graph, np.empty(0)),
                    np.empty((0, 4), np.int64)) for _ in range(self.config.rounds)]
        boxes = torch.as_tensor(window.bboxes, dtype=torch.float32, device=device)
        scale = boxes.new_tensor([window.image_width, window.image_height]*2)
        geometry = torch.cat((boxes/scale, torch.as_tensor(window.confidences,
            dtype=boxes.dtype, device=device)[:, None]), 1)
        appearance = torch.as_tensor(window.appearance, dtype=boxes.dtype, device=device)
        nodes = self.detector(torch.cat((appearance, geometry), 1))
        time = torch.as_tensor(window.frames-window.start_frame, dtype=torch.long, device=device)
        node_tokens = nodes + self.time(time)
        if shared_context:
            # Sequence dimension is the complete detection window.
            nodes = self.context(node_tokens[:, None]).squeeze(1)
        else:
            # Preserve the same encoder parameters and pointwise transform,
            # but make every detection a length-one sequence. Candidate
            # competition below is then the first cross-detection operation.
            nodes = self.context(node_tokens[None]).squeeze(0)
        endpoints = torch.as_tensor(graph.edges, dtype=torch.long, device=device)
        u, v = endpoints.T
        edge_features = torch.as_tensor(graph.features, dtype=nodes.dtype, device=device)
        edges = self.edge_input(torch.cat((nodes[u], nodes[v], edge_features), -1))
        # Initial competition precedes the first actual partition.
        nodes = self._competition(nodes, edges, endpoints, self._gains(nodes, edges, endpoints))
        rounds = []
        context_indices = np.full((len(graph.edges), 4), -1, np.int64)
        for r in range(self.config.rounds):
            gains = self._gains(nodes, edges, endpoints)
            partition = decode_graph(graph, gains.detach().cpu().numpy())
            rounds.append(ConnectionRound(gains, partition, context_indices))
            if r+1 == self.config.rounds or not len(graph.edges):
                continue
            context_indices = conditional_paths(graph, partition, feedback)
            path = self._paths(window, nodes, context_indices)
            selected = torch.as_tensor(partition.selected if feedback else np.zeros(len(edges)),
                dtype=nodes.dtype, device=device)[:, None]
            updated_nodes = self._competition(nodes, edges, endpoints, gains)
            # All edge updates use the SAME previous partition and node snapshot.
            updated_edges = self.edge_update(torch.cat((nodes[u], nodes[v], path, selected), -1), edges)
            nodes, edges = updated_nodes, updated_edges
        return rounds
