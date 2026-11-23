"""Committed BoostTrack motion/appearance evidence for the four-frame decoder.

This is a boundary migration, not the complete BoostTrack++ tracker. Candidate
admission, births, publication and D->D decoding remain with the fixed-lag
framework. Predictions are copies; only oldest-frame commits update state.
"""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ..boosttrack.assoc import iou_batch, MhDist_similarity, shape_similarity_v1
from ..boosttrack.kalmanfilter import KalmanFilter
from ..boosttrack.state import KalmanBoxTracker, convert_bbox_to_z, convert_x_to_bbox
from .boundary import NativeBoTBoundaryCost
from .gates import expanded_legal_mask, unique_mutual_maxima


@dataclass(frozen=True)
class BoostBoundaryConfig:
    # Published defaults; trust_floor is explicitly adapted to the retained
    # candidate floor, since the original 0.6 formula assumes scores >= 0.6.
    trust_floor: float
    iou_threshold: float = 0.3
    lambda_iou: float = 0.5
    lambda_mhd: float = 0.25
    lambda_shape: float = 0.25
    appearance_gate: float = 0.75
    appearance_alpha: float = 0.95

    def __post_init__(self):
        if not 0 <= self.trust_floor < 1:
            raise ValueError("invalid trust floor")

    @property
    def lambda_emb(self):
        return (1 + self.lambda_iou + self.lambda_mhd + self.lambda_shape) * 1.5

    @property
    def gain_divisor(self):
        return 1 + self.lambda_iou + self.lambda_mhd + self.lambda_shape + self.lambda_emb


def boost_evidence(tracks, detections, config):
    """Official association similarity and admissibility, without assignment.

    Returns [detection, track] arrays. The Mahalanobis softmax is computed on
    the whole frame's detection batch, exactly as in the official associate().
    """
    if not tracks or not detections:
        shape = (len(detections), len(tracks))
        return np.zeros(shape), np.zeros(shape, dtype=bool)
    boxes = np.stack([d[0] for d in detections])
    features = np.stack([d[1] for d in detections])
    predicted = np.stack([np.r_[t.get_state()[0], t.get_confidence()] for t in tracks])
    emb = features @ np.stack([t.get_emb() for t in tracks]).T
    iou = iou_batch(boxes, predicted)
    confidence = boxes[:, 4:5] * predicted[None, :, 4]
    confidence[iou < config.iou_threshold] = 0
    z = np.stack([convert_bbox_to_z(b).ravel() for b in boxes])
    means = np.stack([t.kf.x[:4] for t in tracks])
    inv_cov = np.stack([1 / np.diag(t.kf.covariance[:4, :4]) for t in tracks])
    mhd = ((z[:, None, :] - means[None, :, :]) ** 2 * inv_cov[None, :, :]).sum(2)
    similarity = (iou + config.lambda_iou * confidence * iou
                  + config.lambda_mhd * MhDist_similarity(mhd)
                  + config.lambda_shape * confidence * shape_similarity_v1(boxes, predicted)
                  + config.lambda_emb * emb)
    legal = ((iou >= config.iou_threshold)
             | ((iou >= config.iou_threshold / 2) & (emb >= config.appearance_gate)))
    if not np.isfinite(similarity).all():
        raise AssertionError("nonfinite Boost boundary evidence")
    return similarity, legal


class _BoostCost:
    def __init__(self, config, gate):
        if gate not in ('native', 'appearance', 'appearance_native'):
            raise ValueError(f'Unknown candidate gate: {gate}')
        self.config = config
        self.gate = gate

    def _first_association_cost(self, tracks, detections):
        similarity, legal = boost_evidence(tracks, detections, self.config)
        # A fixed analytical conversion, not a fit/dev-selected calibration.
        # Invalid relations are nonpositive and cannot enter a selected path.
        gain = (similarity - self.config.iou_threshold) / self.config.gain_divisor
        if self.gate != 'native' and tracks and detections:
            features = np.stack([item[1] for item in detections])
            embeddings = np.stack([track.get_emb() for track in tracks])
            legal, _ = expanded_legal_mask(legal, features @ embeddings.T, self.config.appearance_gate)
            if self.gate == 'appearance_native':
                legal |= unique_mutual_maxima(gain) & (gain > 0)
        gain = np.where(legal, gain, -1.0)
        return -gain.T


class BoostTrackBoundaryCost(NativeBoTBoundaryCost):
    """Reuse the batch/transaction interface, replace BoT native state/cost."""

    def __init__(self, config: BoostBoundaryConfig, warps, gate='native'):
        self.config = config
        self.oracle = _BoostCost(config, gate)
        self.match_threshold = 0.0
        self.warps = {int(k): np.asarray(v, dtype=np.float64).copy() for k, v in warps.items()}
        self.tracks = {}
        self._allocator = SimpleNamespace(next_id=1)
        self._prediction_cache = {}
        self.statistics = dict(initialized_states=0, updated_states=0,
                               temporary_predictions=0, temporary_cmc_applications=0,
                               recent_costs=0, dormant_costs=0)

    def _detection(self, bbox, confidence, appearance):
        b = np.asarray(bbox, dtype=np.float64).copy()
        b[2:] += b[:2]
        feature = np.asarray(appearance, dtype=np.float32).copy()
        norm = np.linalg.norm(feature)
        if norm <= 1e-12 or not np.isfinite(feature).all():
            raise ValueError("Boost appearance must be nonzero and finite")
        if not self.config.trust_floor - 1e-7 <= confidence <= 1 + 1e-7:
            raise ValueError("confidence outside registered trust range")
        return np.r_[b, confidence], feature / norm

    def initialize_state(self, state, confidence, appearance):
        if state.identity in self.tracks:
            raise AssertionError("duplicate Boost identity")
        bbox, emb = self._detection(state.bbox, confidence, appearance)
        # Match native constructor fields without touching its global ID count.
        track = KalmanBoxTracker.__new__(KalmanBoxTracker)
        track.bbox_to_z_func = convert_bbox_to_z
        track.x_to_bbox_func = convert_x_to_bbox
        track.kf = KalmanFilter(convert_bbox_to_z(bbox))
        track.emb = emb
        track.id = int(state.identity)
        track.time_since_update = track.hit_streak = track.age = 0
        track.frame_id = int(state.last_frame)
        self.tracks[int(state.identity)] = track
        self._prediction_cache.clear()
        self.statistics["initialized_states"] += 1

    def _advance_copy(self, state, frame):
        frame = int(frame)
        if frame <= state.last_frame:
            raise ValueError("prediction must be later than the committed state")
        key = (int(state.identity), int(state.writes), frame)
        if key in self._prediction_cache:
            return self._prediction_cache[key]
        track = deepcopy(self.tracks[int(state.identity)])
        if track.frame_id != state.last_frame:
            raise AssertionError("Boost/public committed clocks diverged")
        for current in range(int(state.last_frame) + 1, frame + 1):
            if current not in self.warps:
                raise KeyError(f"missing shared ECC warp at {current}")
            transform = self.warps[current]
            if transform.shape == (2, 3):
                transform = np.vstack((transform, [0, 0, 1]))
            # Official Boost applies camera_update before predict. Warps are
            # intentionally shared with BoT, rather than recomputed by Boost.
            track.camera_update(transform)
            track.predict()
            track.frame_id = current
            self.statistics["temporary_cmc_applications"] += 1
        self.statistics["temporary_predictions"] += 1
        self._prediction_cache[key] = track
        return track

    def update_state(self, state, frame, bbox, confidence, appearance):
        track = deepcopy(self._advance_copy(state, frame))
        detection, emb = self._detection(bbox, confidence, appearance)
        track.update(detection, float(confidence))
        trust = (float(confidence) - self.config.trust_floor) / (1 - self.config.trust_floor)
        alpha = self.config.appearance_alpha + (1 - self.config.appearance_alpha) * (1 - trust)
        track.update_emb(emb, alpha=alpha)
        track.frame_id = int(frame)
        self.tracks[int(state.identity)] = track
        self._prediction_cache.clear()
        self.statistics["updated_states"] += 1

    def committed_snapshot(self, identity):
        track = self.tracks[int(identity)]
        return dict(frame=int(track.frame_id), mean=track.kf.x.copy(),
                    covariance=track.kf.covariance.copy(), smooth_feature=track.emb.copy(),
                    age=int(track.age), hit_streak=int(track.hit_streak),
                    time_since_update=int(track.time_since_update))
