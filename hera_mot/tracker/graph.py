"""Connection graph representation and decoding from the research tracker."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ConnectionGraph:
    frames: np.ndarray
    edges: np.ndarray
    features: np.ndarray

    def __post_init__(self):
        n, e = len(self.frames), len(self.edges)
        if self.edges.shape != (e, 2) or self.features.shape != (e, 7):
            raise ValueError('Expected edges [E,2], features [E,7]')
        if e and (self.edges.min() < 0 or self.edges.max() >= n):
            raise ValueError('Edge endpoint outside graph')
        if e and not np.all(self.frames[self.edges[:, 0]] < self.frames[self.edges[:, 1]]):
            raise ValueError('Edges must point strictly forward in time')
        if len(set(map(tuple, self.edges.tolist()))) != e:
            raise ValueError('Duplicate candidate edge')
        if not np.isfinite(self.features).all():
            raise ValueError('Nonfinite edge features')

    @property
    def size(self):
        return len(self.frames)


@dataclass(frozen=True)
class Partition:
    paths: tuple
    selected: np.ndarray


def decode_graph(graph, gains):
    """Maximize sum(g*x); each row has its own zero-valued null column.

    Nonpositive edges are omitted, so exact ties with none prefer no connection.
    Hungarian is exact for this round's fixed additive objective, not for the
    nonlinear score changes that a future refinement round might produce.
    """
    gains = np.asarray(gains, np.float64)
    if gains.shape != (len(graph.edges),) or not np.isfinite(gains).all():
        raise ValueError('Expected finite gain per candidate')
    n = graph.size
    selected = np.zeros(len(gains), bool)
    if not n:
        return Partition((), selected)
    cost = np.full((n, 2*n), np.inf)
    cost[np.arange(n), n + np.arange(n)] = 0.
    usable = gains > 0
    u, v = graph.edges[usable].T
    cost[u, v] = -gains[usable]
    rows, cols = linear_sum_assignment(cost)
    lookup = {tuple(edge): k for k, edge in enumerate(graph.edges.tolist())}
    for i, j in zip(rows, cols):
        if j < n:
            selected[lookup[(int(i), int(j))]] = True
    successor, predecessor = {}, {}
    for i, j in graph.edges[selected]:
        successor[int(i)] = int(j)
        predecessor[int(j)] = int(i)
    paths = []
    for start in range(n):
        if start in predecessor:
            continue
        path = [start]
        while path[-1] in successor:
            path.append(successor[path[-1]])
        paths.append(tuple(path))
    assert sum(map(len, paths)) == n
    return Partition(tuple(paths), selected)


def conditional_paths(graph, partition, feedback=True):
    """Discrete indices only; the model gathers live, differentiable features."""
    prefixes, suffixes = {}, {}
    for path in partition.paths:
        for k, i in enumerate(path):
            prefixes[i], suffixes[i] = path[:k+1], path[k:]
    result = np.full((len(graph.edges), 4), -1, np.int64)
    for k, (i, j) in enumerate(graph.edges):
        path = prefixes[i] + suffixes[j] if feedback else (int(i), int(j))
        if len(path) > 4 or np.any(np.diff(graph.frames[list(path)]) <= 0):
            raise ValueError('Invalid conditional path')
        result[k, :len(path)] = path
    return result
