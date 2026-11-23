"""Score detections competing for the same historical identity."""
import numpy as np
import torch
from .history_messages import (
    HistoryContrastGraph, relative_features, bounded_contrast,
)


class HistorySourceContrastGraph(HistoryContrastGraph):
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
        sources = torch.as_tensor(edges[:, 0], device=device)
        gains = np.asarray(history['gains'])
        if gains.shape != (len(edges),) or not np.isfinite(gains).all():
            raise ValueError('history native gain domain')
        if not len(edges):
            return (self.history_score.weight.new_empty((0, self.c4.config.dim)),
                    sources, torch.empty(0, dtype=torch.bool, device=device))
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
        eligible = torch.as_tensor(gains > 0, device=device)
        features, active = relative_features(emb[:len(edges)], sources, eligible, h)
        return features, sources, active

    def history_delta(self, window, graph, history):
        features, sources, active = self.history_features(window, graph, history)
        if not len(features):
            return self.history_score.weight.sum().expand(0)
        return bounded_contrast(features, sources, active, self.history_score.weight, len(history['ids']), self.bound)
