"""Shared recent/dormant path-cover allocation, with zero NEW/HOLD options.

Private zero source nulls mean HOLD. An unanchored path root means zero
NEW if frozen birth conditions hold, otherwise zero suppress/defer. These
zero root options are analytically eliminated from the LAP: labeling them
after decoding does not change the objective or constraints. This baseline
cannot learn a NEW advantage or reject positive continuations by GT.
"""
from typing import Dict, Optional, Sequence
import numpy as np
from .lifecycle import (
    AnchorEdges, FixedLagFourFrameTracker, PlannedDetection, WindowCommitPlan,
    decode_anchored_paths,
)
from .graph import ConnectionGraph
from .observations import DetectionWindow


class UnifiedBoundaryFourFrameTracker(FixedLagFourFrameTracker):
    def _anchor_edges(self, window: DetectionWindow, recent: Sequence[int]) -> AnchorEdges:
        if self.anchor_refiner is not None:
            raise ValueError("this fixed experiment does not use M-C")
        pools = self.pools(window.start_frame)
        if tuple(recent) != pools["recent"]:
            raise ValueError("unexpected recent pool")
        ids = tuple(sorted(pools["recent"] + pools["dormant"]))
        recent_set = set(recent)
        edges, gains = [], []
        for frame in np.unique(window.frames).tolist():
            nodes = np.flatnonzero(window.frames == frame).tolist()
            eligible = [i for i, identity in enumerate(ids)
                        if (identity in recent_set
                            and 1 <= frame - self.states[identity].last_frame <= self.config.short_gap)
                        or (identity not in recent_set and frame == window.start_frame
                            and 1 <= frame - self.states[identity].last_frame <= self.config.keep_gap)]
            if not eligible or not nodes:
                continue
            tracks = [self.scorer._advance_copy(self.states[ids[i]], frame) for i in eligible]
            dets = [self.scorer._detection(window.bboxes[n], float(window.confidences[n]),
                                           window.appearance[n]) for n in nodes]
            costs = np.asarray(self.scorer.oracle._first_association_cost(tracks, dets), dtype=np.float64)
            if costs.shape != (len(eligible), len(nodes)) or not np.isfinite(costs).all():
                raise AssertionError("invalid unified frame batch")
            self.scorer.statistics["recent_costs"] += int(costs.size)
            for row, index in enumerate(eligible):
                for col, node in enumerate(nodes):
                    edges.append((index, node))
                    gains.append(float(self.scorer.match_threshold - costs[row, col]))
        order = sorted(range(len(edges)), key=lambda i: edges[i])
        anchors = AnchorEdges(ids, np.asarray([edges[i] for i in order], dtype=np.int64).reshape(-1, 2),
                              np.asarray([gains[i] for i in order], dtype=np.float64))
        self.last_anchors = anchors  # audit only; never reused at the next window
        return anchors

    def _assign_dormant(self, window, paths, dormant):
        return {}  # all recovery candidates already share the one LAP

    def _describe_competition(self, window, graph, detection_gains, plan):
        anchors = self.last_anchors
        selected_ids = {owner for owner in plan.partition.anchor_by_path if owner is not None}
        recent = set(plan.pools["recent"])
        actions = {item.node: item for item in plan.actions}
        roots = []
        for path in plan.partition.paths:
            node = path[0]
            if int(window.frames[node]) != plan.commit_frame:
                continue
            eligible = (len(path) >= self.config.birth_observations
                        and float(window.confidences[list(path)].max()) >= self.config.new_track_threshold)
            entries = [(anchors.anchor_ids[int(index)], float(gain))
                       for (index, target), gain in zip(anchors.edges, anchors.gains) if target == node]
            entries.sort(key=lambda item: (-item[1], item[0]))
            action = actions.get(node)
            roots.append(dict(uid=int(window.detection_keys[node]), path_uids=[int(window.detection_keys[n]) for n in path],
                              NEW=dict(eligible=bool(eligible), gain=0.0), HOLD=dict(gain=0.0),
                              selected=(action.reason if action else "HOLD_SUPPRESS"),
                              owner=(action.identity if action else None),
                              owner_birth_uid=(int(self.track_ledger[action.identity][0])
                                               if action and action.identity in self.track_ledger else None),
                              candidates=[dict(identity=identity, role="recent" if identity in recent else "dormant",
                                               gain=gain, birth_uid=int(self.track_ledger[identity][0]))
                                          for identity, gain in entries[:5]],
                              positive_recent=int(sum(identity in recent and gain > 0 for identity, gain in entries)),
                              positive_dormant=int(sum(identity not in recent and gain > 0 for identity, gain in entries))))
        return dict(frame=plan.commit_frame, roots=roots,
                    paths=[dict(uids=[int(window.detection_keys[n]) for n in path],
                                anchor=owner)
                           for path, owner in zip(plan.partition.paths, plan.partition.anchor_by_path)],
                    selected_detection_edges=[dict(source=int(window.detection_keys[s]),
                                                   target=int(window.detection_keys[t]), gain=float(g))
                                              for (s, t), g, selected in zip(graph.edges, detection_gains,
                                                                          plan.partition.selected_detection)
                                              if selected],
                    selected_anchor_edges=[dict(identity=int(anchors.anchor_ids[i]),
                                                uid=int(window.detection_keys[n]), gain=float(g))
                                           for (i, n), g, selected in zip(anchors.edges, anchors.gains,
                                                                         plan.partition.selected_anchor)
                                           if selected],
                    recent_count=len(recent), dormant_count=len(plan.pools["dormant"]),
                    selected_anchor_count=len(selected_ids),
                    held_identities=[identity for identity in anchors.anchor_ids if identity not in selected_ids],
                    objective=float(np.asarray(detection_gains)[plan.partition.selected_detection].sum()
                                    + anchors.gains[plan.partition.selected_anchor].sum()))

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
                reason = "recent_anchor" if identity in pools["recent"] else "dormant_recovery"
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
        self.last_competition = self._describe_competition(window, graph, detection_gains, plan)
        return plan
