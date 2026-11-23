"""Deployable OLD/NEW/HOLD scores and losses on the actual full-plan LAP."""
import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


class JointOwnerDecisionHead(nn.Module):
    def __init__(self, feature_dim, mean=None, std=None, hidden=64, correction=1.):
        super().__init__()
        self.feature_dim, self.hidden, self.correction = int(feature_dim), int(hidden), float(correction)
        self.register_buffer('mean', torch.zeros(feature_dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.ones(feature_dim) if std is None else torch.as_tensor(std, dtype=torch.float32))
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, base, current):
        if features.shape[1] != self.feature_dim or not torch.isfinite(features).all():
            raise ValueError('invalid GT-blind features')
        x = ((features - self.mean) / self.std.clamp_min(.05)).clamp(-8., 8.)
        delta = self.correction * torch.tanh(self.network(x).squeeze(1))
        # Provisional roots retain native evidence; every path still competes
        # for the same OLD capacity in the actual deployed LAP.
        return base.to(torch.float64) + delta.to(torch.float64) * current


def full_assignment(sample, values, fixed=None):
    count, width = sample['path_count'], sample['old_count'] + 2 * sample['path_count']
    if not count:
        return np.empty(0, np.int64)
    matrix = np.full((count, width), -np.inf)
    lookup = np.full((count, width), -1, np.int64)
    matrix[sample['rows'], sample['columns']] = values
    lookup[sample['rows'], sample['columns']] = np.arange(len(values))
    if fixed:
        for row, variable in fixed.items():
            col = int(sample['columns'][variable])
            keep = matrix[row, col]
            matrix[row] = -np.inf
            matrix[row, col] = keep
    rr, cc = linear_sum_assignment(-matrix)
    result = lookup[rr, cc]
    if np.any(result < 0) or not np.isfinite(matrix[rr, cc]).all():
        raise AssertionError('infeasible complete owner plan')
    # Preserve the original native null/birth tie and zero-initialization.
    if not fixed and 'reference' in sample:
        reference = sample['reference']
        if float(np.asarray(values)[result].sum() - np.asarray(values)[reference].sum()) <= 1e-10:
            return reference.copy()
    return result
