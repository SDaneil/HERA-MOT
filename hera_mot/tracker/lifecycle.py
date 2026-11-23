"""Causal four-frame primary association with fixed-lag identity commits.

This module deliberately does not call BoT-SORT's framewise assignment.  A
four-frame detection graph and recent identity anchors enter one exact
one-in/one-out decoder.  Only detections in the oldest frame are committed;
the remaining suffix is solved again when the next frame becomes available.

The neural model is kept outside this file.  It supplies a gain for every
detection-to-detection edge.  This separation makes the decoder, lifecycle,
and single-write contract testable without a GPU or labels.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import ConnectionGraph
from .observations import DetectionWindow


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm <= 1e-12 else vector / norm


def _iou_distance(first: np.ndarray, second: np.ndarray) -> float:
    """IoU distance for two TLWH boxes."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return 1.0 if union <= 0 else 1.0 - intersection / union


@dataclass(frozen=True)
class AnchorEdges:
    """Sparse anchor-to-detection gains for the joint path decoder."""

    anchor_ids: Tuple[int, ...]
    edges: np.ndarray
    gains: np.ndarray

    def __post_init__(self):
        edges = np.asarray(self.edges)
        gains = np.asarray(self.gains)
        if edges.ndim != 2 or edges.shape[1:] != (2,) or gains.shape != (len(edges),):
            raise ValueError("anchor edges/gains must have shapes [E,2]/[E]")
        if not np.issubdtype(edges.dtype, np.integer):
            raise ValueError("anchor edge indices must be integers")
        if edges.size and (edges[:, 0].min() < 0 or edges[:, 0].max() >= len(self.anchor_ids)):
            raise ValueError("anchor edge refers to an unknown anchor")
        if len(set(self.anchor_ids)) != len(self.anchor_ids):
            raise ValueError("anchor identities must be unique")
        if len(set(map(tuple, edges.tolist()))) != len(edges):
            raise ValueError("duplicate anchor edge")
        if not np.isfinite(gains).all():
            raise ValueError("anchor gains must be finite")


@dataclass(frozen=True)
class AnchoredPartition:
    paths: Tuple[Tuple[int, ...], ...]
    anchor_by_path: Tuple[Optional[int], ...]
    selected_detection: np.ndarray
    selected_anchor: np.ndarray

    def __post_init__(self):
        if len(self.paths) != len(self.anchor_by_path):
            raise ValueError("one anchor label is required per path")


def decode_anchored_paths(
    graph: ConnectionGraph,
    detection_gains: np.ndarray,
    anchors: AnchorEdges,
) -> AnchoredPartition:
    """Exact additive path cover over detections and immutable source anchors.

    Every detection source and every identity anchor gets a private, zero-gain
    null column.  Real detection columns are shared, which enforces a single
    predecessor across both edge types.  Nonpositive gains are omitted, so a
    tie at zero selects the null/hold option.
    """
    detection_gains = np.asarray(detection_gains, dtype=np.float64)
    if detection_gains.shape != (len(graph.edges),) or not np.isfinite(detection_gains).all():
        raise ValueError("expected one finite gain per detection edge")
    anchor_edges = np.asarray(anchors.edges, dtype=np.int64)
    anchor_gains = np.asarray(anchors.gains, dtype=np.float64)
    n, a = graph.size, len(anchors.anchor_ids)
    if anchor_edges.size and (anchor_edges[:, 1].min() < 0 or anchor_edges[:, 1].max() >= n):
        raise ValueError("anchor edge refers to an unknown detection")

    selected_detection = np.zeros(len(graph.edges), dtype=bool)
    selected_anchor = np.zeros(len(anchor_edges), dtype=bool)
    if not n:
        return AnchoredPartition((), (), selected_detection, selected_anchor)

    rows_count = a + n
    cost = np.full((rows_count, n + rows_count), np.inf, dtype=np.float64)
    cost[np.arange(rows_count), n + np.arange(rows_count)] = 0.0

    edge_lookup = {}
    for index, ((source, target), gain) in enumerate(zip(graph.edges, detection_gains)):
        edge_lookup[(a + int(source), int(target))] = index
        if gain > 0:
            cost[a + int(source), int(target)] = -float(gain)
    anchor_lookup = {}
    for index, ((anchor_index, target), gain) in enumerate(zip(anchor_edges, anchor_gains)):
        key = (int(anchor_index), int(target))
        anchor_lookup[key] = index
        if gain > 0:
            cost[key] = -float(gain)

    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= n:
            continue
        if row < a:
            selected_anchor[anchor_lookup[(row, column)]] = True
        else:
            selected_detection[edge_lookup[(row, column)]] = True

    successor: Dict[int, int] = {}
    predecessor: Dict[int, int] = {}
    for source, target in graph.edges[selected_detection]:
        source, target = int(source), int(target)
        successor[source] = target
        predecessor[target] = source
    anchor_predecessor: Dict[int, int] = {}
    for anchor_index, target in anchor_edges[selected_anchor]:
        anchor_predecessor[int(target)] = int(anchors.anchor_ids[int(anchor_index)])

    paths = []
    owners = []
    for start in range(n):
        if start in predecessor:
            continue
        path = [start]
        while path[-1] in successor:
            path.append(successor[path[-1]])
        paths.append(tuple(path))
        owners.append(anchor_predecessor.get(start))
    if sum(map(len, paths)) != n:
        raise AssertionError("decoder did not produce a detection partition")
    if len([owner for owner in owners if owner is not None]) != int(selected_anchor.sum()):
        raise AssertionError("a selected identity anchor did not start one path")
    return AnchoredPartition(tuple(paths), tuple(owners), selected_detection, selected_anchor)


@dataclass
class IdentityState:
    identity: int
    last_frame: int
    bbox: np.ndarray
    prototype: np.ndarray
    velocity: np.ndarray
    writes: int = 1

    def predicted_bbox(self, frame: int) -> np.ndarray:
        result = np.asarray(self.bbox, dtype=np.float32).copy()
        dt = int(frame) - int(self.last_frame)
        result[:2] += np.asarray(self.velocity, dtype=np.float32) * dt
        return result


@dataclass(frozen=True)
class FrozenLifecycleCost:
    """Frozen, label-free P0 boundary compatibility.

    It follows the registered BoT-SORT distance convention: IoU distance is
    optionally fused with detection confidence, cosine distance is clipped at
    zero, and appearance is disabled outside the proximity gate.  P0 uses a
    constant-velocity committed endpoint instead of mutating a Kalman state.
    That approximation is recorded by the runner and is intentionally not a
    learned component.
    """

    match_threshold: float
    proximity_threshold: float
    appearance_threshold: float
    fuse_score: bool = True

    def initialize_state(
        self,
        state: IdentityState,
        confidence: float,
        appearance: np.ndarray,
    ) -> None:
        """Initialize optional boundary state; this baseline has none."""

    def update_state(
        self,
        state: IdentityState,
        frame: int,
        bbox: np.ndarray,
        confidence: float,
        appearance: np.ndarray,
    ) -> None:
        """Update optional boundary state; this baseline has none."""

    def _appearance_distance(self, prototype: np.ndarray, feature: np.ndarray) -> float:
        cosine = float(_unit(prototype) @ _unit(feature))
        return max(0.0, 1.0 - cosine)

    def recent_gain(
        self,
        state: IdentityState,
        frame: int,
        bbox: np.ndarray,
        confidence: float,
        appearance: np.ndarray,
    ) -> Optional[float]:
        raw_geometry = _iou_distance(state.predicted_bbox(frame), bbox)
        geometry = raw_geometry
        if self.fuse_score:
            geometry = 1.0 - (1.0 - raw_geometry) * float(confidence)
        appearance_cost = self._appearance_distance(state.prototype, appearance)
        if raw_geometry > self.proximity_threshold or appearance_cost > self.appearance_threshold:
            appearance_cost = 1.0
        cost = min(geometry, appearance_cost)
        gain = float(self.match_threshold - cost)
        return gain if np.isfinite(gain) else None

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
        descriptor = _unit(np.asarray(appearances, dtype=np.float32).mean(axis=0))
        appearance_cost = self._appearance_distance(state.prototype, descriptor)
        # Long-gap recovery is appearance-led.  A weak, gap-decayed geometry
        # term only breaks near ties and never makes a failed appearance gate
        # feasible.
        if appearance_cost > self.appearance_threshold:
            return None
        first = int(np.argmin(frames))
        gap = max(1, int(frames[first]) - int(state.last_frame))
        geometry = _iou_distance(state.predicted_bbox(int(frames[first])), boxes[first])
        cost = appearance_cost + 0.05 * (0.5 ** (gap / 4.0)) * geometry
        gain = float(self.appearance_threshold - cost)
        return gain if np.isfinite(gain) else None


class BoundaryCost(Protocol):
    """State and scoring contract shared by frozen and native boundaries."""

    def initialize_state(
        self,
        state: IdentityState,
        confidence: float,
        appearance: np.ndarray,
    ) -> None: ...

    def update_state(
        self,
        state: IdentityState,
        frame: int,
        bbox: np.ndarray,
        confidence: float,
        appearance: np.ndarray,
    ) -> None: ...

    def recent_gain(
        self,
        state: IdentityState,
        frame: int,
        bbox: np.ndarray,
        confidence: float,
        appearance: np.ndarray,
    ) -> Optional[float]: ...

    def dormant_gain(
        self,
        state: IdentityState,
        frames: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,
        appearances: np.ndarray,
    ) -> Optional[float]: ...


class AnchorRefiner(Protocol):
    """Optional label-free residual applied to generated recent-anchor edges."""

    def refine(
        self,
        states: Sequence[IdentityState],
        window: DetectionWindow,
        anchors: AnchorEdges,
    ) -> AnchorEdges: ...


@dataclass(frozen=True)
class FixedLagConfig:
    short_gap: int = 3
    keep_gap: int = 40
    admission_threshold: float = 0.1
    new_track_threshold: float = 0.6
    birth_observations: int = 2
    prototype_momentum: float = 0.9
    enable_complete_four_frame_low_only_birth: bool = False

    def __post_init__(self):
        if self.short_gap < 1 or self.keep_gap <= self.short_gap:
            raise ValueError("keep_gap must be larger than short_gap")
        if self.birth_observations < 1:
            raise ValueError("birth_observations must be positive")
        if not 0 <= self.prototype_momentum < 1:
            raise ValueError("prototype_momentum must lie in [0,1)")
        thresholds = (
            self.admission_threshold,
            self.new_track_threshold,
        )
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in thresholds):
            raise ValueError("detection thresholds must lie in [0,1]")


@dataclass(frozen=True)
class CommittedDetection:
    frame: int
    detection_key: int
    identity: int
    bbox: np.ndarray
    confidence: float
    reason: str
    path_keys: Tuple[int, ...]


@dataclass(frozen=True)
class CommitResult:
    committed: Tuple[CommittedDetection, ...]
    suppressed_keys: Tuple[int, ...]
    partition: AnchoredPartition
    pools: Dict[str, Tuple[int, ...]]


@dataclass(frozen=True)
class PlannedDetection:
    """One oldest-frame write selected by a read-only window plan."""

    node: int
    frame: int
    detection_key: int
    identity: int
    reason: str
    path_keys: Tuple[int, ...]
    bbox: np.ndarray
    confidence: float
    appearance: np.ndarray


@dataclass(frozen=True)
class ProvisionalPlan:
    """Audit-only suffix left after the oldest frame is contracted.

    This object is deliberately never consumed by candidate generation or
    scoring.  A future window is always decoded from its observations and the
    committed identity state again.
    """

    window_frame_ids: Tuple[int, ...]
    detection_nodes: Tuple[Tuple[int, int], ...]
    selected_predecessors: Tuple[Tuple[int, int], ...]
    selected_successors: Tuple[Tuple[int, int], ...]
    root_entry_edges: Tuple[Tuple[int, int], ...]
    temporary_owner: Tuple[Tuple[int, Optional[int]], ...]
    rootless_components: Tuple[Tuple[int, ...], ...]
    plan_version: int
    source_state_version: int


@dataclass(frozen=True)
class WindowCommitPlan:
    """Validated intent for one atomic oldest-frame commit."""

    commit_frame: int
    source_state_version: int
    source_next_identity: int
    next_identity_after: int
    actions: Tuple[PlannedDetection, ...]
    suppressed_keys: Tuple[int, ...]
    partition: AnchoredPartition
    pools: Dict[str, Tuple[int, ...]]
    newly_removed: Tuple[int, ...]
    provisional_plan: ProvisionalPlan


@dataclass(frozen=True)
class TrackerStateSnapshot:
    """In-memory checkpoint of committed state; model weights are not copied."""

    states: Dict[int, IdentityState]
    removed: frozenset
    ledger: Dict[int, int]
    track_ledger: Dict[int, Tuple[int, ...]]
    birth_frames: Dict[int, int]
    next_identity: int
    next_commit_frame: int
    state_version: int
    statistics: Dict[str, int]
    provisional_plan: Optional[ProvisionalPlan]
    scorer_state: Any


class FixedLagFourFrameTracker:
    """P0 state machine for causal fixed-lag physical and identity tracking."""

    def __init__(
        self,
        config: FixedLagConfig,
        scorer: BoundaryCost,
        anchor_refiner: Optional[AnchorRefiner] = None,
    ):
        self.config = config
        self.scorer = scorer
        self.anchor_refiner = anchor_refiner
        self.states: Dict[int, IdentityState] = {}
        self.removed = set()
        self.ledger: Dict[int, int] = {}
        self.track_ledger: Dict[int, list[int]] = {}
        self.birth_frames: Dict[int, int] = {}
        self.next_identity = 1
        self.next_commit_frame = 1
        self.state_version = 0
        self.last_provisional_plan: Optional[ProvisionalPlan] = None
        self.statistics: Dict[str, int] = {}

    def _increment(self, key: str, amount: int = 1) -> None:
        self.statistics[key] = self.statistics.get(key, 0) + int(amount)

    @staticmethod
    def _is_new_identity_reason(reason: str) -> bool:
        return reason in {"new", "four_frame_low_only"}

    @staticmethod
    def _is_complete_current_four_frame_path(
        window: DetectionWindow,
        path: Sequence[int],
        commit_frame: int,
    ) -> bool:
        """Return whether a selected path contains one real node in c..c+3.

        The suffix is intentionally checked against the current window rather
        than a saved plan.  In particular, the exception cannot fire during a
        one-to-three-frame end-of-sequence flush.
        """
        if int(window.end_frame) < int(commit_frame) + 3 or len(path) != 4:
            return False
        nodes = np.asarray(path, dtype=np.int64)
        frames = tuple(int(value) for value in window.frames[nodes])
        keys = tuple(int(value) for value in window.detection_keys[nodes])
        return (
            frames == tuple(range(int(commit_frame), int(commit_frame) + 4))
            and len(set(keys)) == 4
        )

    def pools(self, commit_frame: int) -> Dict[str, Tuple[int, ...]]:
        """Classify identities without mutating their persistent lifecycle."""
        recent, dormant, removed = [], [], []
        for identity, state in sorted(self.states.items()):
            gap = int(commit_frame) - int(state.last_frame)
            if identity in self.removed or gap > self.config.keep_gap:
                removed.append(identity)
            elif gap <= self.config.short_gap:
                recent.append(identity)
            else:
                dormant.append(identity)
        return {"recent": tuple(recent), "dormant": tuple(dormant), "removed": tuple(removed)}

    def _scorer_snapshot(self, identities: Optional[Sequence[int]] = None) -> Any:
        snapshot = getattr(self.scorer, "snapshot_state", None)
        if snapshot is not None:
            if identities is None:
                return ("hook", snapshot())
            try:
                return ("hook", snapshot(identities=tuple(map(int, identities))))
            except TypeError:
                return ("hook", snapshot())
        # FrozenLifecycleCost and test doubles are small.  Stateful production
        # boundaries provide the hook above, avoiding copies of model objects.
        return ("dict", deepcopy(getattr(self.scorer, "__dict__", {})))

    def _restore_scorer(self, token: Any) -> None:
        kind, value = token
        if kind == "hook":
            restore = getattr(self.scorer, "restore_state", None)
            if restore is None:
                raise RuntimeError("scorer snapshot hook has no matching restore hook")
            restore(value)
            return
        if kind != "dict":
            raise RuntimeError("unknown scorer snapshot kind")
        state = getattr(self.scorer, "__dict__", None)
        if state is None:
            if value:
                raise RuntimeError("cannot restore scorer without mutable state")
            return
        state.clear()
        state.update(deepcopy(value))

    def snapshot_state(self) -> TrackerStateSnapshot:
        """Capture committed state for deterministic in-memory save/restore."""
        return TrackerStateSnapshot(
            states=deepcopy(self.states),
            removed=frozenset(self.removed),
            ledger=dict(self.ledger),
            track_ledger={
                int(identity): tuple(map(int, keys))
                for identity, keys in self.track_ledger.items()
            },
            birth_frames=dict(self.birth_frames),
            next_identity=int(self.next_identity),
            next_commit_frame=int(self.next_commit_frame),
            state_version=int(self.state_version),
            statistics=dict(self.statistics),
            provisional_plan=deepcopy(self.last_provisional_plan),
            scorer_state=self._scorer_snapshot(),
        )

    def restore_state(self, snapshot: TrackerStateSnapshot) -> None:
        """Restore a checkpoint without changing configuration or model weights."""
        if not isinstance(snapshot, TrackerStateSnapshot):
            raise TypeError("expected TrackerStateSnapshot")
        self.states = deepcopy(snapshot.states)
        self.removed = set(snapshot.removed)
        self.ledger = dict(snapshot.ledger)
        self.track_ledger = {
            int(identity): list(map(int, keys))
            for identity, keys in snapshot.track_ledger.items()
        }
        self.birth_frames = dict(snapshot.birth_frames)
        self.next_identity = int(snapshot.next_identity)
        self.next_commit_frame = int(snapshot.next_commit_frame)
        self.state_version = int(snapshot.state_version)
        self.statistics = dict(snapshot.statistics)
        self.last_provisional_plan = deepcopy(snapshot.provisional_plan)
        self._restore_scorer(snapshot.scorer_state)

    def _transaction_snapshot(self, plan: WindowCommitPlan) -> dict:
        action_identities = sorted({int(action.identity) for action in plan.actions})
        identities = sorted({
            int(action.identity)
            for action in plan.actions
            if not self._is_new_identity_reason(action.reason)
        })
        return {
            "states": {
                identity: deepcopy(self.states[identity]) for identity in identities
            },
            "existing_state_ids": frozenset(self.states),
            "removed": set(self.removed),
            "track_ledger": {
                identity: (
                    identity in self.track_ledger,
                    list(self.track_ledger.get(identity, ())),
                )
                for identity in identities
            },
            "birth_frames": {
                identity: (
                    identity in self.birth_frames,
                    self.birth_frames.get(identity),
                )
                for identity in identities
            },
            "next_identity": int(self.next_identity),
            "next_commit_frame": int(self.next_commit_frame),
            "state_version": int(self.state_version),
            "statistics": dict(self.statistics),
            "provisional_plan": self.last_provisional_plan,
            "scorer_state": self._scorer_snapshot(action_identities),
        }

    def _rollback_transaction(self, token: dict, plan: WindowCommitPlan) -> None:
        existing = token["existing_state_ids"]
        for identity in list(self.states):
            if identity not in existing:
                del self.states[identity]
        self.states.update(deepcopy(token["states"]))
        self.removed = set(token["removed"])
        for action in plan.actions:
            self.ledger.pop(int(action.detection_key), None)
        for identity, (present, keys) in token["track_ledger"].items():
            if present:
                self.track_ledger[identity] = list(keys)
            else:
                self.track_ledger.pop(identity, None)
        for identity in list(self.track_ledger):
            if identity not in existing:
                self.track_ledger.pop(identity, None)
        for identity, (present, frame) in token["birth_frames"].items():
            if present:
                self.birth_frames[identity] = int(frame)
            else:
                self.birth_frames.pop(identity, None)
        for identity in list(self.birth_frames):
            if identity not in existing:
                self.birth_frames.pop(identity, None)
        self.next_identity = int(token["next_identity"])
        self.next_commit_frame = int(token["next_commit_frame"])
        self.state_version = int(token["state_version"])
        self.statistics = dict(token["statistics"])
        self.last_provisional_plan = token["provisional_plan"]
        self._restore_scorer(token["scorer_state"])

    def _anchor_edges(self, window: DetectionWindow, recent: Sequence[int]) -> AnchorEdges:
        batch = getattr(self.scorer, "recent_candidates", None)
        if batch is not None:
            edges, gains = batch(
                tuple(self.states[int(identity)] for identity in recent),
                window,
                self.config.short_gap,
            )
            anchors = AnchorEdges(
                tuple(map(int, recent)),
                np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                np.asarray(gains, dtype=np.float64),
            )
        else:
            edges, gains = [], []
            for anchor_index, identity in enumerate(recent):
                state = self.states[int(identity)]
                for detection_index, frame in enumerate(window.frames.tolist()):
                    gap = int(frame) - int(state.last_frame)
                    if not 1 <= gap <= self.config.short_gap:
                        continue
                    gain = self.scorer.recent_gain(
                        state,
                        int(frame),
                        window.bboxes[detection_index],
                        float(window.confidences[detection_index]),
                        window.appearance[detection_index],
                    )
                    if gain is not None:
                        edges.append((anchor_index, detection_index))
                        gains.append(gain)
            anchors = AnchorEdges(
                tuple(map(int, recent)),
                np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                np.asarray(gains, dtype=np.float64),
            )
        if self.anchor_refiner is None:
            return anchors
        refined = self.anchor_refiner.refine(
            tuple(self.states[int(identity)] for identity in recent),
            window,
            anchors,
        )
        if refined.anchor_ids != anchors.anchor_ids or not np.array_equal(
            refined.edges, anchors.edges
        ):
            raise ValueError("anchor refiner may change scores but not candidate edges")
        before = np.asarray(anchors.gains) > 0
        after = np.asarray(refined.gains) > 0
        self._increment("refined_anchor_edges", len(anchors.edges))
        self._increment("anchor_edges_activated", int((~before & after).sum()))
        self._increment("anchor_edges_deactivated", int((before & ~after).sum()))
        return refined

    def _assign_dormant(
        self,
        window: DetectionWindow,
        paths: Sequence[Tuple[int, ...]],
        dormant: Sequence[int],
    ) -> Dict[int, int]:
        """Return path-position -> dormant identity using one global LAP."""
        if not paths or not dormant:
            return {}
        row_count, identity_count = len(paths), len(dormant)
        utility = np.full((row_count, identity_count + row_count), -np.inf, dtype=np.float64)
        utility[np.arange(row_count), identity_count + np.arange(row_count)] = 0.0
        batch = getattr(self.scorer, "dormant_gain_matrix", None)
        if batch is not None:
            gains = np.asarray(
                batch(
                    tuple(self.states[int(identity)] for identity in dormant),
                    window,
                    tuple(paths),
                ),
                dtype=np.float64,
            )
            if gains.shape != (row_count, identity_count) or not np.isfinite(gains).all():
                raise ValueError("invalid batched dormant gain matrix")
            utility[:, :identity_count] = np.where(gains > 0, gains, -np.inf)
        else:
            for row, path in enumerate(paths):
                indices = np.asarray(path, dtype=np.int64)
                for column, identity in enumerate(dormant):
                    gain = self.scorer.dormant_gain(
                        self.states[int(identity)],
                        window.frames[indices],
                        window.bboxes[indices],
                        window.confidences[indices],
                        window.appearance[indices],
                    )
                    if gain is not None and gain > 0:
                        utility[row, column] = gain
        rows, columns = linear_sum_assignment(-utility)
        result = {}
        for row, column in zip(rows.tolist(), columns.tolist()):
            if column < identity_count and utility[row, column] > 0:
                result[int(row)] = int(dormant[column])
        return result

    def _create_state(
        self,
        frame: int,
        bbox: np.ndarray,
        confidence: float,
        appearance: np.ndarray,
        expected_identity: Optional[int] = None,
    ) -> int:
        identity = self.next_identity
        if expected_identity is not None and identity != int(expected_identity):
            raise AssertionError("planned identity allocation no longer matches tracker state")
        self.next_identity += 1
        state = IdentityState(
            identity=identity,
            last_frame=int(frame),
            bbox=np.asarray(bbox, dtype=np.float32).copy(),
            prototype=_unit(appearance),
            velocity=np.zeros(2, dtype=np.float32),
        )
        self.scorer.initialize_state(state, float(confidence), appearance)
        self.states[identity] = state
        return identity

    def _update_state(
        self,
        identity: int,
        frame: int,
        bbox: np.ndarray,
        confidence: float,
        appearance: np.ndarray,
    ) -> None:
        state = self.states[int(identity)]
        if int(frame) <= state.last_frame:
            raise AssertionError("persistent identity state may only advance in real time")
        self.scorer.update_state(
            state,
            int(frame),
            np.asarray(bbox, dtype=np.float32),
            float(confidence),
            appearance,
        )
        old_center = state.bbox[:2] + state.bbox[2:] / 2.0
        bbox = np.asarray(bbox, dtype=np.float32).copy()
        new_center = bbox[:2] + bbox[2:] / 2.0
        state.velocity = (new_center - old_center) / float(int(frame) - state.last_frame)
        state.bbox = bbox
        state.last_frame = int(frame)
        momentum = self.config.prototype_momentum
        state.prototype = _unit(momentum * state.prototype + (1.0 - momentum) * _unit(appearance))
        state.writes += 1

    def _validate_window_inputs(
        self,
        window: DetectionWindow,
        graph: ConnectionGraph,
        detection_gains: np.ndarray,
        commit_frame: Optional[int] = None,
    ) -> int:
        commit_frame = window.start_frame if commit_frame is None else int(commit_frame)
        if commit_frame != self.next_commit_frame:
            raise ValueError("commit cursor must advance by exactly one real frame")
        if commit_frame != window.start_frame:
            raise ValueError("the committed frame must be the oldest frame in the window")
        if window.window_size > 4 or not np.array_equal(graph.frames, window.frames):
            raise ValueError("graph/window mismatch or window longer than four frames")
        if np.any(window.confidences < self.config.admission_threshold):
            raise ValueError("window contains detections below the registered admission threshold")
        gains = np.asarray(detection_gains)
        if gains.shape != (len(graph.edges),) or not np.isfinite(gains).all():
            raise ValueError("expected one finite gain per detection edge")
        return commit_frame

    def _suffix_plan(
        self,
        window: DetectionWindow,
        graph: ConnectionGraph,
        partition: AnchoredPartition,
        commit_frame: int,
        owner_by_path: Dict[int, int],
    ) -> ProvisionalPlan:
        """Materialize the uncommitted suffix for audit, never for inference."""
        kept = {
            int(index)
            for index, frame in enumerate(window.frames.tolist())
            if int(frame) > int(commit_frame)
        }
        key = lambda node: int(window.detection_keys[int(node)])
        selected_pairs = [
            (key(source), key(target))
            for selected, (source, target) in zip(
                partition.selected_detection.tolist(), graph.edges.tolist()
            )
            if selected and int(source) in kept and int(target) in kept
        ]
        root_entries = []
        temporary_owner = []
        rootless = []
        for path_index, path in enumerate(partition.paths):
            suffix = tuple(int(node) for node in path if int(node) in kept)
            if not suffix:
                continue
            owner = owner_by_path.get(path_index)
            if owner is None:
                rootless.append(tuple(key(node) for node in suffix))
            else:
                root_entries.append((int(owner), key(suffix[0])))
            temporary_owner.extend((key(node), owner) for node in suffix)
        frames = tuple(sorted({int(frame) for frame in window.frames.tolist() if frame > commit_frame}))
        nodes = tuple(
            (key(index), int(window.frames[index])) for index in sorted(kept)
        )
        pairs = tuple(selected_pairs)
        return ProvisionalPlan(
            window_frame_ids=frames,
            detection_nodes=nodes,
            selected_predecessors=pairs,
            selected_successors=pairs,
            root_entry_edges=tuple(root_entries),
            temporary_owner=tuple(temporary_owner),
            rootless_components=tuple(rootless),
            plan_version=int(self.state_version) + 1,
            source_state_version=int(self.state_version),
        )

    def plan_window(
        self,
        window: DetectionWindow,
        graph: ConnectionGraph,
        detection_gains: np.ndarray,
        commit_frame: Optional[int] = None,
    ) -> WindowCommitPlan:
        """Decode a window and build a commit intent without persistent writes."""
        commit_frame = self._validate_window_inputs(
            window, graph, detection_gains, commit_frame
        )

        pools = self.pools(commit_frame)
        anchor_edges = self._anchor_edges(window, pools["recent"])
        partition = decode_anchored_paths(graph, detection_gains, anchor_edges)
        path_for_node = {
            node: (path_index, path)
            for path_index, path in enumerate(partition.paths)
            for node in path
        }
        oldest_nodes = np.flatnonzero(window.frames == commit_frame).tolist()
        if len(oldest_nodes) != len(set(oldest_nodes)):
            raise AssertionError("duplicate oldest-frame detection")
        oldest_keys = [int(window.detection_keys[node]) for node in oldest_nodes]
        repeated_keys = sorted(set(oldest_keys) & set(self.ledger))
        if repeated_keys:
            raise AssertionError(
                "a detection was committed more than once: "
                + ",".join(map(str, repeated_keys))
            )

        unresolved_paths = []
        unresolved_path_indices = []
        direct_identity: Dict[int, int] = {}
        for node in oldest_nodes:
            path_index, path = path_for_node[int(node)]
            owner = partition.anchor_by_path[path_index]
            if owner is None:
                unresolved_path_indices.append(path_index)
                unresolved_paths.append(path)
            else:
                direct_identity[path_index] = int(owner)

        dormant_local = self._assign_dormant(window, unresolved_paths, pools["dormant"])
        dormant_identity = {
            unresolved_path_indices[row]: identity for row, identity in dormant_local.items()
        }

        actions = []
        suppressed = []
        frame_identities = set()
        owner_by_path = {
            int(path_index): int(owner)
            for path_index, owner in enumerate(partition.anchor_by_path)
            if owner is not None
        }
        planned_next_identity = int(self.next_identity)
        for node in oldest_nodes:
            path_index, path = path_for_node[int(node)]
            if partition.anchor_by_path[path_index] is not None:
                identity = direct_identity[path_index]
                reason = "recent_anchor"
            elif path_index in dormant_identity:
                identity = dormant_identity[path_index]
                reason = "dormant_recovery"
            else:
                path_nodes = np.asarray(path, dtype=np.int64)
                path_max_confidence = float(window.confidences[path_nodes].max())
                eligible = (
                    len(path) >= self.config.birth_observations
                    and path_max_confidence >= self.config.new_track_threshold
                )
                reason = "new"
                if (
                    not eligible
                    and self.config.enable_complete_four_frame_low_only_birth
                    and len(path) >= self.config.birth_observations
                    and path_max_confidence < self.config.new_track_threshold
                    and self._is_complete_current_four_frame_path(
                        window, path, commit_frame
                    )
                ):
                    eligible = True
                    reason = "four_frame_low_only"
                if not eligible:
                    key = int(window.detection_keys[node])
                    suppressed.append(key)
                    continue
                identity = planned_next_identity
                planned_next_identity += 1

            key = int(window.detection_keys[node])
            if identity in frame_identities:
                raise AssertionError("one identity was assigned twice in the same committed frame")
            frame_identities.add(identity)
            owner_by_path[path_index] = int(identity)
            actions.append(
                PlannedDetection(
                    node=int(node),
                    frame=commit_frame,
                    detection_key=key,
                    identity=identity,
                    bbox=np.asarray(window.bboxes[node], dtype=np.float32).copy(),
                    confidence=float(window.confidences[node]),
                    appearance=np.asarray(window.appearance[node], dtype=np.float32).copy(),
                    reason=reason,
                    path_keys=tuple(int(window.detection_keys[index]) for index in path),
                )
            )

        plan = WindowCommitPlan(
            commit_frame=int(commit_frame),
            source_state_version=int(self.state_version),
            source_next_identity=int(self.next_identity),
            next_identity_after=int(planned_next_identity),
            actions=tuple(actions),
            suppressed_keys=tuple(suppressed),
            partition=partition,
            pools={name: tuple(values) for name, values in pools.items()},
            newly_removed=tuple(
                identity for identity in pools["removed"] if identity not in self.removed
            ),
            provisional_plan=self._suffix_plan(
                window, graph, partition, commit_frame, owner_by_path
            ),
        )
        self._validate_plan(plan, window)
        return plan

    def _validate_plan(self, plan: WindowCommitPlan, window: DetectionWindow) -> None:
        if plan.source_state_version != self.state_version:
            raise RuntimeError("stale commit plan: persistent state version changed")
        if plan.commit_frame != self.next_commit_frame:
            raise RuntimeError("stale commit plan: commit cursor changed")
        if plan.source_next_identity != self.next_identity:
            raise RuntimeError("stale commit plan: identity allocator changed")
        if plan.commit_frame != window.start_frame:
            raise ValueError("commit plan/window mismatch")
        current_pools = self.pools(plan.commit_frame)
        if current_pools != plan.pools:
            raise RuntimeError("stale commit plan: lifecycle pools changed")
        expected_removed = tuple(
            identity for identity in plan.pools["removed"] if identity not in self.removed
        )
        if plan.newly_removed != expected_removed:
            raise AssertionError("commit plan has an invalid lifecycle transition")
        if plan.provisional_plan.source_state_version != plan.source_state_version:
            raise AssertionError("provisional and commit plan versions diverged")
        oldest = {
            int(window.detection_keys[index])
            for index in np.flatnonzero(window.frames == plan.commit_frame).tolist()
        }
        action_keys = [int(action.detection_key) for action in plan.actions]
        if len(action_keys) != len(set(action_keys)):
            raise AssertionError("commit plan contains duplicate detections")
        if set(action_keys) | set(plan.suppressed_keys) != oldest:
            raise AssertionError("commit plan does not cover the complete oldest frame")
        if set(action_keys) & set(plan.suppressed_keys):
            raise AssertionError("a detection cannot be both committed and suppressed")
        repeated = sorted(set(action_keys) & set(self.ledger))
        if repeated:
            raise AssertionError(
                "a detection was committed more than once: " + ",".join(map(str, repeated))
            )
        identities = [int(action.identity) for action in plan.actions]
        if len(identities) != len(set(identities)):
            raise AssertionError("one identity was assigned twice in the same committed frame")
        expected_new = list(range(plan.source_next_identity, plan.next_identity_after))
        actual_new = [
            int(action.identity)
            for action in plan.actions
            if self._is_new_identity_reason(action.reason)
        ]
        if actual_new != expected_new:
            raise AssertionError("new identities are not a contiguous deterministic allocation")
        for action in plan.actions:
            if action.frame != plan.commit_frame or action.detection_key not in action.path_keys:
                raise AssertionError("invalid planned detection metadata")
            if not 0 <= action.node < len(window.frames):
                raise AssertionError("planned detection node is outside the window")
            if (
                int(window.frames[action.node]) != action.frame
                or int(window.detection_keys[action.node]) != action.detection_key
                or float(window.confidences[action.node]) != action.confidence
                or not np.array_equal(window.bboxes[action.node], action.bbox)
                or not np.array_equal(window.appearance[action.node], action.appearance)
            ):
                raise AssertionError("planned detection no longer matches the source window")
            path = next(
                path for path in plan.partition.paths if int(action.node) in path
            )
            expected_path_keys = tuple(
                int(window.detection_keys[index]) for index in path
            )
            if action.path_keys != expected_path_keys:
                raise AssertionError("planned path no longer matches the decoded partition")
            if action.reason == "recent_anchor" and action.identity not in plan.pools["recent"]:
                raise AssertionError("recent action does not refer to a recent identity")
            if action.reason == "dormant_recovery" and action.identity not in plan.pools["dormant"]:
                raise AssertionError("dormant action does not refer to a dormant identity")
            if self._is_new_identity_reason(action.reason) and action.identity in self.states:
                raise AssertionError("new identity allocation collides with an existing track")
            if action.reason not in {
                "recent_anchor", "dormant_recovery", "new", "four_frame_low_only"
            }:
                raise AssertionError("unknown commit reason")

    def _validate_ledger_registry(self, updated: Sequence[PlannedDetection]) -> None:
        # The previous state passed this invariant and every new key was
        # rejected if it already existed.  It is therefore sufficient to
        # validate the append delta plus the aggregate count; scanning every
        # historical observation at every frame would make long videos O(N^2).
        if sum(map(len, self.track_ledger.values())) != len(self.ledger):
            raise AssertionError("global and per-track committed ledgers diverged")
        for action in updated:
            identity, key = int(action.identity), int(action.detection_key)
            if identity not in self.states:
                raise AssertionError("track ledger refers to a missing permanent identity")
            if self.ledger.get(key) != identity:
                raise AssertionError("global ledger has the wrong permanent identity")
            keys = self.track_ledger.get(identity)
            if not keys or keys[-1] != key:
                raise AssertionError("per-track committed ledger append is invalid")
        if any(identity not in self.states for identity in self.birth_frames):
            raise AssertionError("birth registry refers to a missing permanent identity")

    def commit_plan(
        self,
        plan: WindowCommitPlan,
        window: DetectionWindow,
    ) -> CommitResult:
        """Atomically apply a previously built plan after full validation."""
        self._validate_plan(plan, window)
        transaction = self._transaction_snapshot(plan)
        committed = []
        try:
            self.removed.update(plan.newly_removed)
            for action in plan.actions:
                if self._is_new_identity_reason(action.reason):
                    identity = self._create_state(
                        action.frame,
                        action.bbox,
                        action.confidence,
                        action.appearance,
                        expected_identity=action.identity,
                    )
                    self.birth_frames[identity] = int(action.frame)
                    self.track_ledger[identity] = []
                    self._increment("new_identity_commits")
                    if action.reason == "four_frame_low_only":
                        self._increment("four_frame_low_only_birth_commits")
                else:
                    identity = int(action.identity)
                    self.birth_frames.setdefault(identity, int(self.states[identity].last_frame))
                    self.track_ledger.setdefault(identity, [])
                    self._update_state(
                        identity,
                        action.frame,
                        action.bbox,
                        action.confidence,
                        action.appearance,
                    )
                    if action.reason == "recent_anchor":
                        self._increment("recent_anchor_commits")
                    else:
                        self._increment("dormant_recovery_commits")
                self.ledger[action.detection_key] = identity
                self.track_ledger[identity].append(int(action.detection_key))
                committed.append(
                    CommittedDetection(
                        frame=action.frame,
                        detection_key=action.detection_key,
                        identity=identity,
                        bbox=np.asarray(action.bbox, dtype=np.float32).copy(),
                        confidence=action.confidence,
                        reason=action.reason,
                        path_keys=action.path_keys,
                    )
                )
            for key in plan.suppressed_keys:
                self._increment("suppressed_oldest")
                path = next(
                    path for path in plan.partition.paths
                    if int(key) in {
                        int(window.detection_keys[index]) for index in path
                    }
                )
                if len(path) < self.config.birth_observations:
                    self._increment("suppressed_short_path")
                else:
                    self._increment("suppressed_low_confidence_path")
            if self.next_identity != plan.next_identity_after:
                raise AssertionError("identity allocator did not reach planned state")
            self.next_commit_frame += 1
            self.state_version += 1
            self.last_provisional_plan = plan.provisional_plan
            self._increment("committed_frames")
            self._increment("committed_detections", len(committed))
            self._increment("selected_detection_edges", int(plan.partition.selected_detection.sum()))
            self._increment("selected_anchor_edges", int(plan.partition.selected_anchor.sum()))
            self._validate_ledger_registry(plan.actions)
        except BaseException:
            self._rollback_transaction(transaction, plan)
            raise
        return CommitResult(
            tuple(committed), plan.suppressed_keys, plan.partition, plan.pools
        )

    def commit_window(
        self,
        window: DetectionWindow,
        graph: ConnectionGraph,
        detection_gains: np.ndarray,
        commit_frame: Optional[int] = None,
    ) -> CommitResult:
        """Plan, validate, then atomically commit the oldest available frame."""
        plan = self.plan_window(window, graph, detection_gains, commit_frame)
        return self.commit_plan(plan, window)
