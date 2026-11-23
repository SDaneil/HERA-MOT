"""Structured association loss for fixed, independently computed action utilities."""

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from .models.ias import full_assignment


def _utility_face(sample, energies, utilities, utility_best):
    rows, columns = sample["rows"], sample["columns"]
    row_count, old_count = sample["path_count"], sample["old_count"]
    positions = np.arange(len(energies))
    is_old = columns < old_count
    row_eq = coo_matrix((np.ones(len(rows)), (rows, positions)), shape=(row_count, len(rows)))
    old_capacity = coo_matrix(
        (np.ones(is_old.sum()), (columns[is_old], positions[is_old])),
        shape=(old_count, len(rows)),
    ).tocsc()

    # Subtract a constant per row using its unit-sum constraint. Scaling
    # preserves the face while keeping small utility differences visible to HiGHS.
    row_max = np.full(row_count, -np.inf)
    np.maximum.at(row_max, rows, utilities)
    face = (utilities - row_max[rows]) * 1e4
    equalities = vstack((row_eq, coo_matrix(face[None, :])), format="csc")
    bounds = dict(
        A_eq=equalities, b_eq=np.r_[np.ones(row_count), face[utility_best].sum()],
        A_ub=old_capacity, b_ub=np.ones(old_count), bounds=(0., 1.),
    )
    options = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
    result = linprog(-1e4 * energies, **bounds, method="highs-ds", options=options)
    if result.status == 2:
        result = linprog(-1e4 * energies, **bounds, method="highs-ds",
                         options={**options, "presolve": False})
    if not result.success or result.x is None:
        raise RuntimeError(f"Utility-optimal assignment LP failed: {result.message}")
    if np.max(np.abs(result.x - result.x.round())) > 1e-6:
        raise RuntimeError("Utility-optimal assignment LP returned a fractional solution.")

    chosen = np.flatnonzero(result.x > .5)
    chosen = chosen[np.argsort(rows[chosen])]
    if not np.array_equal(rows[chosen], np.arange(row_count)) or len(np.unique(columns[chosen])) != row_count:
        raise RuntimeError("Utility-optimal comparator violates assignment capacity.")
    if abs(utilities[chosen].sum() - utilities[utility_best].sum()) > 1e-8:
        raise RuntimeError("Comparator lies outside the utility-optimal face.")
    return chosen


def structured_loss(
    sample, scores, utilities, known_rows, old_residuals, *, margin=0.2, residual_weight=0.01
):
    """Return the scalar structured hinge plus the mean squared OLD residual.

    Rows enumerate detections and columns enumerate shared OLD followed by
    row-private NEW and HOLD. Scores and utilities follow the flattened legal
    actions. Unsupervised rows have zero utility and still compete for OLD
    capacity. Residuals contain one value per legal OLD action.
    """
    row_count, old_count = sample["path_count"], sample["old_count"]
    rows, columns = np.asarray(sample["rows"]), np.asarray(sample["columns"])
    known_rows = np.asarray(known_rows)
    if (row_count < 0 or old_count < 0 or known_rows.shape != (row_count,)
            or (known_rows.size and known_rows.dtype != np.bool_)):
        raise ValueError("Expected a boolean supervision mask with one entry per detection.")
    if (scores.ndim != 1 or not scores.is_floating_point()
            or rows.shape != scores.shape or columns.shape != scores.shape):
        raise ValueError("Expected one floating-point score per flattened legal action.")
    if len(rows) and (rows.dtype.kind not in "iu" or columns.dtype.kind not in "iu"):
        raise ValueError("Action row and column indices must be integers.")
    rows, columns = rows.astype(np.int64, copy=False), columns.astype(np.int64, copy=False)
    is_old = columns < old_count
    private = (columns == old_count + rows) | (columns == old_count + row_count + rows)
    action_pairs = np.stack((rows, columns), axis=1)
    if (np.any((rows < 0) | (rows >= row_count)) or np.any(columns < 0)
            or not np.all(is_old | private) or len(np.unique(rows)) != row_count
            or len(np.unique(action_pairs, axis=0)) != len(rows)):
        raise ValueError("Expected unique legal actions in the shared OLD and row-private column layout.")
    if (old_residuals.shape != (int(is_old.sum()),) or old_residuals.device != scores.device
            or not old_residuals.is_floating_point() or not torch.isfinite(old_residuals).all()):
        raise ValueError("Expected one finite OLD residual per legal OLD action on the scores device.")
    if not (np.isfinite(margin) and np.isfinite(residual_weight) and margin >= 0 and residual_weight >= 0):
        raise ValueError("Margin and residual weight must be finite and nonnegative.")

    utilities = np.asarray(utilities, dtype=np.float64)
    energies = scores.detach().double().cpu().numpy()
    if utilities.shape != energies.shape or not np.isfinite(utilities).all() or not np.isfinite(energies).all():
        raise ValueError("Expected one finite score and utility per legal action.")
    residual = old_residuals.square().mean() if old_residuals.numel() else old_residuals.sum()
    penalty = residual_weight * residual
    count = int(known_rows.sum())
    if count == 0:
        return scores[:0].double().sum() + penalty

    pure = dict(path_count=row_count, old_count=old_count, rows=rows, columns=columns)
    augmented_sample = pure
    if "reference" in sample:
        reference = np.asarray(sample["reference"])
        if (reference.shape != (row_count,) or reference.dtype.kind not in "iu"
                or np.any((reference < 0) | (reference >= len(rows)))
                or not np.array_equal(rows[reference], np.arange(row_count))
                or len(np.unique(columns[reference])) != row_count):
            raise ValueError("Expected a complete, feasible native reference assignment.")
        augmented_sample = {**pure, "reference": reference}
    utilities = np.where(known_rows[rows], utilities, 0.)
    utility_best = full_assignment(pure, utilities)
    predicted = full_assignment(augmented_sample, energies - margin * utilities)
    comparator = _utility_face(pure, energies, utilities, utility_best)
    regret = max(0., float(utilities[utility_best].sum() - utilities[predicted].sum()))

    predicted = torch.as_tensor(predicted, device=scores.device)
    comparator = torch.as_tensor(comparator, device=scores.device)
    hinge = scores[predicted].double().sum() - scores[comparator].double().sum() + margin * regret - 1e-10
    return hinge.clamp_min(0.) / count + penalty
