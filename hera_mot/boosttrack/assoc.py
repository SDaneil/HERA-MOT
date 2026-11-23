"""Native association cues from BoostTrack. See LICENSE."""
from copy import deepcopy

import numpy as np


def shape_similarity_v1(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(-(np.abs(dw - tw)/np.maximum(dw, tw) + np.abs(dh - th)/np.maximum(dw, tw)))


def MhDist_similarity(mahalanobis_distance: np.ndarray, softmax_temp: float = 1.0) -> np.ndarray:
    limit = 13.2767  # 99% conf interval https://www.mathworks.com/help/stats/chi2inv.html
    mahalanobis_distance = deepcopy(mahalanobis_distance)
    mask = mahalanobis_distance > limit
    mahalanobis_distance[mask] = limit
    mahalanobis_distance = limit - mahalanobis_distance

    mahalanobis_distance = np.exp(mahalanobis_distance/softmax_temp) / np.exp(mahalanobis_distance/softmax_temp).sum(0).reshape((1, -1))
    mahalanobis_distance = np.where(mask, 0, mahalanobis_distance)
    return mahalanobis_distance


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    return o
