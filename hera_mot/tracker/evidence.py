"""GT-blind, owner-conditioned evidence for whole four-frame hypotheses.

Each path has a private native Kalman/appearance branch. Hypothetical updates
change its next observation likelihood, never the persistent tracker or its
prediction cache. This module only builds evidence; it does not select paths,
publish detections, replace native states, or constitute an admitted experiment.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

from .boost import (
    BoostTrackBoundaryCost, KalmanBoxTracker, KalmanFilter,
    convert_bbox_to_z, convert_x_to_bbox,
)
from .actions import BlindObservationStore


TOKEN_NAMES = (
    'window_offset', 'observation_gap', 'confidence',
    'cosine_committed_smooth', 'cosine_hypothetical_smooth',
    'cosine_last_committed', 'cosine_birth_committed',
    'cosine_history_mean', 'cosine_history_max', 'cosine_history_min',
    'innovation_cx', 'innovation_cy', 'innovation_height', 'innovation_aspect',
    'log1p_mahalanobis', 'relative_logdet', 'NEW_comparison_valid',
    'OLD_minus_NEW_NLL', 'OLD_minus_NEW_logdet',
    'cosine_path_prefix_mean', 'cosine_path_prefix_min',
    'committed_age', 'committed_history_length',
)


@dataclass(frozen=True)
class ConditionalPathEvidence:
    paths: tuple
    owners: tuple
    decision_frame: int
    tokens: np.ndarray
    valid: np.ndarray
    predicted_mean: np.ndarray
    innovation_covariance: np.ndarray
    mahalanobis: np.ndarray
    negative_log_likelihood: np.ndarray
    new_negative_log_likelihood: np.ndarray
    new_comparison_valid: np.ndarray


def _advance_private(track, target_frame, warps):
    if target_frame < track.frame_id:
        raise ValueError('private branch cannot travel backward')
    for frame in range(track.frame_id + 1, int(target_frame) + 1):
        transform = np.asarray(warps[frame], np.float64)
        if transform.shape == (2, 3):
            transform = np.vstack((transform, [0., 0., 1.]))
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError('invalid actual ECC transform')
        track.camera_update(transform)
        track.predict()
        track.frame_id = frame


def _innovation(track, detection):
    mean, covariance = track.kf.project()
    observation = convert_bbox_to_z(detection).ravel()
    factor = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(factor, observation - mean)
    distance = float(whitened @ whitened)
    logdet = float(2 * np.log(np.diag(factor)).sum())
    nll = .5 * (distance + logdet + 4 * np.log(2 * np.pi))
    if not np.isfinite(nll):
        raise ValueError('nonfinite conditional native likelihood')
    return mean, covariance, whitened, distance, logdet, nll


def _hypothetical_update(track, detection, appearance, confidence, config):
    track.update(detection, confidence)
    trust = (confidence - config.trust_floor) / (1 - config.trust_floor)
    alpha = config.appearance_alpha + (1 - config.appearance_alpha) * (1 - trust)
    track.update_emb(appearance, alpha=alpha)


def _new_branch(detection, appearance, frame):
    # Native constructor fields without changing KalmanBoxTracker.count.
    track = KalmanBoxTracker.__new__(KalmanBoxTracker)
    track.bbox_to_z_func, track.x_to_bbox_func = convert_bbox_to_z, convert_x_to_bbox
    track.kf = KalmanFilter(convert_bbox_to_z(detection))
    track.emb, track.id = appearance.copy(), -1
    track.time_since_update = track.hit_streak = track.age = 0
    track.frame_id = int(frame)
    return track


def conditional_path_evidence(tracker, store, decision_frame, paths, owners):
    """Build OLD evidence for arbitrary actual candidate paths, including future roots.

    The caller supplies legal path/owner pairs from its frozen candidate domain.
    All inputs are immutable pre-commit state, raw observations and actual ECC.
    There is no current-only score mask. A NEW branch is initialized *at* its
    root observation, so its root has no predictive density to compare.
    """
    if type(tracker.scorer) is not BoostTrackBoundaryCost:
        raise TypeError('conditional evidence requires the original native Boost boundary')
    if not isinstance(store, BlindObservationStore):
        raise TypeError('conditional evidence requires the GT-whitelisted observation store')
    decision_frame = int(decision_frame)
    paths = tuple(tuple(int(k) for k in path) for path in paths)
    owners = tuple(int(i) for i in owners)
    if len(paths) != len(owners):
        raise ValueError('one committed owner per candidate path is required')
    count = len(paths)
    tokens = np.zeros((count, 4, len(TOKEN_NAMES)), np.float32)
    valid = np.zeros((count, 4), bool)
    means = np.zeros((count, 4, 4), np.float64)
    covariances = np.zeros((count, 4, 4, 4), np.float64)
    distances = np.zeros((count, 4), np.float64)
    nll = np.zeros((count, 4), np.float64)
    new_nll = np.zeros((count, 4), np.float64)
    new_valid = np.zeros((count, 4), bool)
    scorer = tracker.scorer
    prepared, galleries = {}, {}
    for row, (path, owner) in enumerate(zip(paths, owners)):
        if not path or len(path) > 4 or len(set(path)) != len(path):
            raise ValueError('one to four distinct actual observations are required')
        indices = store.indices(path)
        frames = store.frames[indices]
        if (frames[0] < decision_frame or frames[-1] > decision_frame + 3
                or np.any(np.diff(frames) <= 0)):
            raise ValueError('path must stay within the actual ordered t..t+3 window')
        state = tracker.states[owner]
        committed = scorer.tracks[owner]
        history = tuple(int(k) for k in tracker.track_ledger[owner])
        if not history or state.last_frame >= decision_frame or committed.frame_id != state.last_frame:
            raise ValueError('owner must be committed strictly before the decision frame')
        hi = store.indices(history)
        if (np.any(store.frames[hi] >= decision_frame)
                or np.any(np.diff(store.frames[hi]) <= 0)
                or store.frames[hi[-1]] != state.last_frame):
            raise ValueError('identity history includes uncommitted or inconsistent observations')
        if owner not in galleries:
            # Same deterministic coverage as S3; direct full-dimensional dot products.
            selected = sorted(set(range(min(4, len(history)))) |
                set(np.linspace(0, len(history) - 1, min(32, len(history))).astype(int).tolist()) |
                set(range(max(0, len(history) - 8), len(history))))
            galleries[owner] = store.appearance[hi[selected]], store.appearance[hi[-1]], store.appearance[hi[0]]
        gallery, last, birth = galleries[owner]
        root_frame = int(frames[0])
        key = (owner, root_frame)
        if key not in prepared:
            initial = deepcopy(committed)
            _advance_private(initial, root_frame, scorer.warps)
            prepared[key] = initial
        # Never update the shared prepared branch or scorer's cached prediction.
        branch = deepcopy(prepared[key])
        fresh = None
        previous = int(state.last_frame)
        prefix = []
        for step, (index, frame) in enumerate(zip(indices, frames)):
            frame = int(frame)
            _advance_private(branch, frame, scorer.warps)
            confidence = float(store.scores[index])
            detection, appearance = scorer._detection(store.boxes[index], confidence, store.appearance[index])
            mean, covariance, whitened, distance, logdet, value = _innovation(branch, detection)
            means[row, step], covariances[row, step] = mean, covariance
            distances[row, step], nll[row, step] = distance, value
            residual_nll = residual_logdet = 0.
            if fresh is not None:
                _advance_private(fresh, frame, scorer.warps)
                *_, fresh_logdet, fresh_value = _innovation(fresh, detection)
                new_nll[row, step], new_valid[row, step] = fresh_value, True
                residual_nll, residual_logdet = value - fresh_value, logdet - fresh_logdet
                _hypothetical_update(fresh, detection, appearance, confidence, scorer.config)
            else:
                fresh = _new_branch(detection, appearance, frame)
            cosine_history = gallery @ appearance
            cosine_prefix = np.asarray(prefix) @ appearance if prefix else np.array([0.])
            noise = branch.kf.cov_update_policy.get_R(branch.kf.x, 0)
            noise_sign, noise_logdet = np.linalg.slogdet(noise)
            if noise_sign != 1:
                raise ValueError('invalid native measurement noise')
            tokens[row, step] = np.r_[
                (frame - decision_frame) / 3., np.log1p(frame - previous) / 8., confidence,
                float(committed.emb @ appearance), float(branch.emb @ appearance),
                float(last @ appearance), float(birth @ appearance),
                cosine_history.mean(), cosine_history.max(), cosine_history.min(),
                np.arcsinh(whitened) / 10., np.log1p(distance) / 10., (logdet - noise_logdet) / 10.,
                float(new_valid[row, step]), np.arcsinh(residual_nll) / 10., residual_logdet / 10.,
                cosine_prefix.mean(), cosine_prefix.min(),
                np.log1p(decision_frame - int(store.frames[hi[0]])) / 8., np.log1p(len(history)) / 8.]
            valid[row, step] = True
            _hypothetical_update(branch, detection, appearance, confidence, scorer.config)
            prefix.append(appearance)
            previous = frame
    arrays = (tokens, valid, means, covariances, distances, nll, new_nll, new_valid)
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite conditional evidence packet')
    for a in arrays:
        a.setflags(write=False)
    return ConditionalPathEvidence(paths, owners, decision_frame, *arrays)
