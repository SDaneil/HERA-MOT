"""Read GT-free observations and bind them to the selected-frame clock."""

from pathlib import Path

import numpy as np


FEATURE_DIM = 3584
OBSERVATION_FIELDS = {"keys", "frames", "boxes", "scores", "width", "height", "length"}
EMBEDDING_FIELDS = {"graph_appearance", "secondary_appearance"}


def _positive_integer(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu" or array.item() < 1:
        raise ValueError(f"{name} must be a positive integer scalar")
    return int(array.item())


def _embedding_array(value, count, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (count, FEATURE_DIM) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape ({count}, {FEATURE_DIM}) with finite values")
    squared_norm = np.einsum("ij,ij->i", array, array)
    if not np.allclose(squared_norm, 1, rtol=0, atol=0.005):
        raise ValueError(f"{name} must contain L2-normalized embeddings")
    return array


def load_observations(path, *, primary=None, secondary=None):
    """Load one sequence. External NPY descriptors may replace NPZ fields."""
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        if not OBSERVATION_FIELDS <= names or not names <= OBSERVATION_FIELDS | EMBEDDING_FIELDS:
            raise ValueError("input NPZ must contain the documented observation fields only")
        raw = {field: data[field] for field in OBSERVATION_FIELDS}
        arrays = {}
        for name, external in (("graph_appearance", primary), ("secondary_appearance", secondary)):
            if external is not None:
                if Path(external).suffix.lower() != ".npy":
                    raise ValueError(f"external {name} must be an NPY file")
                arrays[name] = np.load(external, mmap_mode="r", allow_pickle=False)
            elif name in names:
                arrays[name] = data[name]
            else:
                raise ValueError(f"missing {name}")

    for name in ("width", "height", "length"):
        raw[name] = _positive_integer(raw[name], name)
    keys, frames = raw["keys"], raw["frames"]
    if keys.ndim != 1 or frames.shape != keys.shape or keys.dtype.kind not in "iu" or frames.dtype.kind not in "iu":
        raise ValueError("keys and frames must be aligned one-dimensional integer arrays")
    count = len(keys)
    if len(np.unique(keys)) != count or np.any(keys < 0) or np.any(keys > np.iinfo(np.int64).max):
        raise ValueError("keys must be unique nonnegative int64 values")
    if np.any(frames < 1) or np.any(frames > raw["length"]):
        raise ValueError("frames must be one-based within the declared sequence length")
    raw["keys"], raw["frames"] = keys.astype(np.int64), frames.astype(np.int64)
    boxes = np.asarray(raw["boxes"], dtype=np.float32)
    scores = np.asarray(raw["scores"], dtype=np.float32)
    if boxes.shape != (count, 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError("boxes must be finite TLWH rows with positive widths and heights")
    if scores.shape != (count,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must contain one finite confidence in [0, 1] per detection")
    raw["boxes"], raw["scores"] = boxes, scores
    raw["graph_appearance"] = _embedding_array(arrays["graph_appearance"], count, "graph_appearance")
    auxiliary = _embedding_array(arrays["secondary_appearance"], count, "secondary_appearance")
    return raw, auxiliary


def sample_observations(raw, secondary, stride):
    """Keep frame 1 and every stride-th frame, including empty sampled steps."""
    source_frames = np.arange(1, raw["length"] + 1, stride, dtype=np.int64)
    selected = np.flatnonzero((raw["frames"] - 1) % stride == 0)
    order = np.lexsort((raw["keys"][selected], raw["frames"][selected]))
    rows = selected[order]
    selection = rows
    if len(rows) == len(raw["keys"]) and np.array_equal(rows, np.arange(len(rows))):
        selection = slice(None)
    sampled = {key: raw[key][selection] for key in ("keys", "frames", "boxes", "scores", "graph_appearance")}
    sampled["frames"] = (sampled["frames"] - 1) // stride + 1
    sampled.update(width=raw["width"], height=raw["height"], length=len(source_frames))
    return sampled, secondary[selection], source_frames


def load_warps(path, source_frames):
    """Read transforms from each previous selected image to the current one."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"warps", "source_pairs"}:
            raise ValueError("warp NPZ must contain warps and source_pairs")
        matrices, pairs = data["warps"], data["source_pairs"]
    count = len(source_frames)
    expected_pairs = np.column_stack((np.r_[source_frames[0], source_frames[:-1]], source_frames))
    if pairs.dtype.kind not in "iu" or not np.array_equal(pairs, expected_pairs):
        raise ValueError("each warp must use the declared pair of consecutive selected source images")
    if matrices.shape not in ((count, 2, 3), (count, 3, 3)) or matrices.dtype.kind != "f":
        raise ValueError("warps must be floating point arrays of shape [steps, 2, 3] or [steps, 3, 3]")
    if not np.isfinite(matrices).all():
        raise ValueError("warps contain nonfinite values")
    homogeneous = matrices
    if matrices.shape[1] == 2:
        bottom = np.broadcast_to([0, 0, 1], (count, 1, 3))
        homogeneous = np.concatenate((matrices, bottom), axis=1)
    if not np.array_equal(homogeneous[0], np.eye(3)) or np.any(np.abs(np.linalg.det(homogeneous)) < 1e-12):
        raise ValueError("the first warp must be identity and all warps must be nonsingular")
    return dict(enumerate(matrices, start=1))


def identity_warps(length):
    return {step: np.eye(3, dtype=np.float64) for step in range(1, length + 1)}
