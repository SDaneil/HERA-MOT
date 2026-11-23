"""Reciprocal appearance and positive native-gain gates."""
import numbers

import numpy as np


def expanded_legal_mask(strict_legal, cosines, appearance_gate):
    """Return (expanded, added) masks in [detections,tracks] full-batch order."""
    legal = np.asarray(strict_legal)
    cosine = np.asarray(cosines)
    if legal.ndim != 2 or legal.dtype != np.bool_ or cosine.shape != legal.shape:
        raise ValueError('aligned boolean legal and cosine matrices [detections,tracks] required')
    if not np.issubdtype(cosine.dtype, np.floating) or not np.isfinite(cosine).all():
        raise ValueError('finite floating whole-frame cosine matrix required')
    if isinstance(appearance_gate, (bool, np.bool_)) or not isinstance(appearance_gate, numbers.Real) or not np.isfinite(appearance_gate) or not -1 <= appearance_gate <= 1:
        raise ValueError('existing finite appearance cosine gate within[-1,1] required')
    if not legal.size:
        return legal.copy(), np.zeros_like(legal)
    row_max = cosine.max(axis=1, keepdims=True)
    column_max = cosine.max(axis=0, keepdims=True)
    row_top = cosine == row_max
    column_top = cosine == column_max
    reciprocal_unique = (row_top & column_top & (row_top.sum(axis=1, keepdims=True) == 1)
                         & (column_top.sum(axis=0, keepdims=True) == 1)
                         & (cosine >= appearance_gate))
    added = reciprocal_unique & ~legal
    return legal | added, added


def unique_mutual_maxima(values):
    """Full-matrix exact unique row AND column maxima; ties never add edges."""
    values=np.asarray(values)
    if values.ndim!=2 or not np.issubdtype(values.dtype,np.floating) or not np.isfinite(values).all():
        raise ValueError('Expected a finite floating-point score matrix')
    if not values.size:return np.zeros(values.shape,dtype=bool)
    row=values==values.max(axis=1,keepdims=True)
    column=values==values.max(axis=0,keepdims=True)
    return row & column & (row.sum(axis=1,keepdims=True)==1) & (column.sum(axis=0,keepdims=True)==1)
