"""Shared boundary transaction and candidate batch operations."""
from copy import deepcopy
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .lifecycle import IdentityState
from .observations import DetectionWindow


class NativeBoTBoundaryCost:
    """Native batch and commit interface specialized by the Boost boundary."""

    def snapshot_state(self, identities: Optional[Sequence[int]] = None) -> dict:
        """Capture only mutable committed/cache state, never model weights."""
        selected = (
            sorted(self.tracks)
            if identities is None
            else sorted({int(identity) for identity in identities})
        )
        return {
            "partial": identities is not None,
            "tracks": {
                identity: deepcopy(self.tracks[identity])
                for identity in selected
                if identity in self.tracks
            },
            "selected": tuple(selected),
            "allocator_next_id": int(self._allocator.next_id),
            "statistics": dict(self.statistics),
        }

    def restore_state(self, snapshot: Mapping) -> None:
        """Restore a transaction or explicit tracker checkpoint."""
        if snapshot["partial"]:
            for identity in snapshot["selected"]:
                self.tracks.pop(int(identity), None)
            self.tracks.update(deepcopy(snapshot["tracks"]))
        else:
            self.tracks = deepcopy(snapshot["tracks"])
        self._allocator.next_id = int(snapshot["allocator_next_id"])
        # Cache contents never define committed state; recomputation is exact.
        self._prediction_cache.clear()
        self.statistics = dict(snapshot["statistics"])

    def _native_cost(
        self,
        state: IdentityState,
        frame: int,
        bbox: np.ndarray,
        confidence: float,
        appearance: np.ndarray,
    ) -> float:
        predicted = self._advance_copy(state, int(frame))
        detection = self._detection(bbox, confidence, appearance)
        cost = self.oracle._first_association_cost([predicted], [detection])
        if cost.shape != (1, 1) or not np.isfinite(cost[0, 0]):
            raise AssertionError("native boundary cost is invalid")
        return float(cost[0, 0])

    def recent_gain(
        self,
        state: IdentityState,
        frame: int,
        bbox: np.ndarray,
        confidence: float,
        appearance: np.ndarray,
    ) -> Optional[float]:
        cost = self._native_cost(state, frame, bbox, confidence, appearance)
        self.statistics["recent_costs"] += 1
        return self.match_threshold - cost

    def recent_candidates(
        self,
        states: Sequence[IdentityState],
        window: DetectionWindow,
        short_gap: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized native C0 for all legal recent-anchor edges."""
        edges = []
        gains = []
        for frame in np.unique(window.frames).tolist():
            detection_indices = np.flatnonzero(window.frames == int(frame))
            anchor_indices = [
                index
                for index, state in enumerate(states)
                if 1 <= int(frame) - int(state.last_frame) <= int(short_gap)
            ]
            if not len(detection_indices) or not anchor_indices:
                continue
            tracks = [self._advance_copy(states[index], int(frame)) for index in anchor_indices]
            detections = [
                self._detection(
                    window.bboxes[index],
                    float(window.confidences[index]),
                    window.appearance[index],
                )
                for index in detection_indices.tolist()
            ]
            costs = np.asarray(
                self.oracle._first_association_cost(tracks, detections),
                dtype=np.float64,
            )
            expected = (len(anchor_indices), len(detection_indices))
            if costs.shape != expected or not np.isfinite(costs).all():
                raise AssertionError("invalid batched native recent cost")
            self.statistics["recent_costs"] += int(costs.size)
            for local_anchor, anchor_index in enumerate(anchor_indices):
                for local_detection, detection_index in enumerate(detection_indices.tolist()):
                    edges.append((anchor_index, detection_index))
                    gains.append(self.match_threshold - float(costs[local_anchor, local_detection]))
        if edges:
            order = sorted(range(len(edges)), key=lambda index: edges[index])
            edges = [edges[index] for index in order]
            gains = [gains[index] for index in order]
        return (
            np.asarray(edges, dtype=np.int64).reshape(-1, 2),
            np.asarray(gains, dtype=np.float64),
        )

    def dormant_gain(
        self,
        state: IdentityState,
        frames: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,
        appearances: np.ndarray,
    ) -> Optional[float]:
        if not len(frames):
            return None
        first = int(np.argmin(frames))
        cost = self._native_cost(
            state,
            int(frames[first]),
            boxes[first],
            float(confidences[first]),
            appearances[first],
        )
        self.statistics["dormant_costs"] += 1
        return self.match_threshold - cost

    def dormant_gain_matrix(
        self,
        states: Sequence[IdentityState],
        window: DetectionWindow,
        paths: Sequence[Tuple[int, ...]],
    ) -> np.ndarray:
        """Return [path, identity] gains using each path's oldest observation."""
        result = np.empty((len(paths), len(states)), dtype=np.float64)
        if not len(paths) or not len(states):
            return result
        first_indices = [
            int(path[int(np.argmin(window.frames[np.asarray(path, dtype=np.int64)]))])
            for path in paths
        ]
        target_frames = {int(window.frames[index]) for index in first_indices}
        if len(target_frames) != 1:
            raise ValueError("dormant paths must all contain the committed oldest frame")
        frame = target_frames.pop()
        tracks = [self._advance_copy(state, frame) for state in states]
        detections = [
            self._detection(
                window.bboxes[index],
                float(window.confidences[index]),
                window.appearance[index],
            )
            for index in first_indices
        ]
        costs = np.asarray(
            self.oracle._first_association_cost(tracks, detections),
            dtype=np.float64,
        )
        expected = (len(states), len(paths))
        if costs.shape != expected or not np.isfinite(costs).all():
            raise AssertionError("invalid batched native dormant cost")
        self.statistics["dormant_costs"] += int(costs.size)
        result[:] = (self.match_threshold - costs).T
        return result
