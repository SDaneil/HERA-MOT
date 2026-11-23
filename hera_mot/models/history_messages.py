"""Normalize candidate features and compute bounded, zero-sum score corrections."""
import numpy as np
import torch
from torch import nn
from .history_graph import HistoryNodeGraph


def relative_features(embeddings, targets, eligible, node_count, epsilon=1e-4):
    if embeddings.ndim != 2 or targets.shape != eligible.shape or len(targets) != len(embeddings):
        raise ValueError('history contrast domain')
    if targets.dtype != torch.long or eligible.dtype != torch.bool:
        raise ValueError('history contrast indices/mask')
    if not torch.isfinite(embeddings).all():
        raise ValueError('nonfinite history embeddings')
    if len(targets) and (int(targets.min()) < 0 or int(targets.max()) >= node_count):
        raise ValueError('history contrast target')
    width = embeddings.shape[1]
    selected = torch.where(eligible[:, None], embeddings, 0.)
    counts = embeddings.new_zeros(node_count).index_add(0, targets, eligible.to(embeddings.dtype))
    means = embeddings.new_zeros((node_count, width)).index_add(0, targets, selected)
    means = means / counts.clamp_min(1)[:, None]
    active = eligible & (counts[targets] > 1)
    centered = torch.where(active[:, None], embeddings - means[targets], 0.)
    power = embeddings.new_zeros(node_count).index_add(0, targets, centered.square().sum(-1))
    scale = (power / (counts.clamp_min(1) * width) + epsilon**2).sqrt()
    return centered / scale[targets, None], active


def bounded_contrast(features, targets, active, weight, node_count, bound=.25):
    raw = torch.nn.functional.linear(features, weight).squeeze(-1)
    raw = torch.where(active, raw, 0.)
    # A common divisor retains zero-sum contrasts. In the unit box this is
    # exactly linear, permitting a checked, parameterized capacity oracle.
    maximum = raw.new_zeros(node_count).scatter_reduce(0, targets, raw.abs(), reduce='amax', include_self=True)
    return bound * raw / maximum.clamp_min(1)[targets]


class HistoryContrastGraph(HistoryNodeGraph):
    def __init__(self, c4, s3, pixel):
        super().__init__(c4, s3, pixel)
        self.history_score = nn.Linear(c4.config.dim, 1, bias=False)
        nn.init.zeros_(self.history_score.weight)

    def history_features(self, window, graph, history):
        device = next(self.parameters()).device
        h, n = len(history['ids']), graph.size
        edges = np.asarray(history['edges'], np.int64)
        frames = np.asarray(history['frames'])
        if edges.shape != (len(edges), 2):
            raise ValueError('history edge shape')
        if len(frames) != h or np.any(frames >= window.start_frame):
            raise ValueError('uncommitted history node')
        if len(set(map(int, history['ids']))) != h:
            raise ValueError('duplicate history identity')
        if len(edges) and (edges[:, 0].min() < 0 or edges[:, 0].max() >= h
                           or edges[:, 1].min() < 0 or edges[:, 1].max() >= n):
            raise ValueError('history endpoint outside visible graph')
        if len(set(map(tuple, edges.tolist()))) != len(edges):
            raise ValueError('duplicate history edge')
        targets = torch.as_tensor(edges[:, 1], device=device)
        gains = np.asarray(history['gains'])
        if gains.shape != (len(edges),) or not np.isfinite(gains).all():
            raise ValueError('history native gain domain')
        if not len(edges):
            return (self.history_score.weight.new_empty((0, self.c4.config.dim)),
                    targets, torch.empty(0, dtype=torch.bool, device=device))
        boxes = np.concatenate((history['boxes'], window.bboxes))
        scores = np.r_[history['scores'], window.confidences]
        appearance = np.concatenate((history['appearance'], window.appearance))
        x = torch.as_tensor(np.c_[appearance, boxes / np.array([window.image_width, window.image_height] * 2), scores],
                            dtype=torch.float32, device=device)
        offsets = np.r_[np.zeros(h, np.int64), window.frames - window.start_frame]
        nodes = self.c4.context((self.c4.detector(x) + self.c4.time(torch.as_tensor(offsets, device=device)))[None]).squeeze(0)
        hh, dd = edges + np.array([0, h]), np.asarray(graph.edges) + h
        endpoints = torch.as_tensor(np.concatenate((hh, dd)), device=device)
        feature = torch.as_tensor(np.concatenate((history['features'], graph.features)), dtype=nodes.dtype, device=device)
        u, v = endpoints.T
        emb = self.c4.edge_input(torch.cat((nodes[u], nodes[v], feature), -1))
        for _ in range(self.c4.config.rounds):
            gain = self.c4._gains(nodes, emb, endpoints)
            updated = self.c4._competition(nodes, emb, endpoints, gain)
            evidence = (nodes[u] + nodes[v]) * .5
            emb = self.c4.edge_update(torch.cat((nodes[u], nodes[v], evidence,
                                                (gain > 0).to(nodes.dtype)[:, None]), -1), emb)
            nodes = updated
        # Only positive-native OLD alternatives enter the deployed assignment.
        eligible = torch.as_tensor(gains > 0, device=device)
        features, active = relative_features(emb[:len(edges)], targets, eligible, n)
        return features, targets, active

    def history_delta(self, window, graph, history):
        features, targets, active = self.history_features(window, graph, history)
        if not len(features):
            return self.history_score.weight.sum().expand(0)
        return bounded_contrast(features, targets, active, self.history_score.weight, graph.size, self.bound)
