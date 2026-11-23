"""Finite complete-cover joint path/owner policy, with native-prior calibration.

Each proposal is an actual disjoint cover of all observed nodes. Frozen S3
competition features are rebuilt on that cover, never on overlapping path rows.
Owner-conditioned prefix evidence can change both paths and owner assignment.
This engineering implementation has no formal training/performance admission.
"""
from dataclasses import dataclass, replace
import numpy as np
import torch

from ..models.ias import full_assignment
from .evidence import conditional_path_evidence
from .lifecycle import PlannedDetection, decode_anchored_paths
from .actions import (
    JointOwnerDecisionTracker, extract_features, sample_from_plan, materialize_owner_plan,
)


@dataclass(frozen=True)
class CoverProposalConfig:
    max_covers: int = 8
    max_cut_edges: int = 16

    def __post_init__(self):
        if self.max_covers < 1 or self.max_cut_edges < 0:
            raise ValueError('invalid complete-cover proposal budget')


@dataclass(frozen=True)
class ConditionalCover:
    template: object
    sample: dict
    s3_scores: np.ndarray
    base_choice: np.ndarray
    native_score: float
    pair_indices: np.ndarray

    @property
    def frozen_owner_optimum(self):
        return float(self.s3_scores[self.base_choice].sum())


def complete_cover_proposals(graph, gains, anchors, native_partition, config):
    """Fixed one-edge-forbid alternatives; not all paths or k-best enumeration.

    Every alternative is solved as a complete additive anchored LAP. Lowest
    selected DD gains receive the bounded counterfactual attempts. The original
    cover is always first; duplicate physical covers are removed label-free.
    """
    gains = np.asarray(gains, np.float64)
    original_score = float(gains[native_partition.selected_detection].sum()
                           + anchors.gains[native_partition.selected_anchor].sum())
    output = [(native_partition, original_score)]
    if config.max_covers == 1:
        return tuple(output)
    seen = {native_partition.paths}
    attempts = sorted(np.flatnonzero(native_partition.selected_detection).tolist(),
                      key=lambda i: (gains[i], tuple(graph.edges[i])))[:config.max_cut_edges]
    alternatives = []
    for index in attempts:
        modified = gains.copy(); modified[index] = -1.
        partition = decode_anchored_paths(graph, modified, anchors)
        if partition.paths in seen:
            continue
        seen.add(partition.paths)
        score = float(gains[partition.selected_detection].sum()
                      + anchors.gains[partition.selected_anchor].sum())
        if score > original_score + 1e-9:
            raise AssertionError('original native LAP was not an optimum')
        alternatives.append((partition, score))
    alternatives.sort(key=lambda x: (-x[1], x[0].paths))
    output.extend(alternatives[:config.max_covers - 1])
    return tuple(output)


def template_for_cover(actor, window, graph, original, partition):
    mapping = {node: row for row, path in enumerate(partition.paths) for node in path}
    actions, suppressed = [], []
    cursor = actor.next_identity
    owner_by_path = {r: owner for r, owner in enumerate(partition.anchor_by_path) if owner is not None}
    for node in np.flatnonzero(window.frames == original.commit_frame).tolist():
        row = mapping[node]; path = partition.paths[row]
        owner = partition.anchor_by_path[row]
        if owner is not None:
            identity = int(owner)
            reason = 'recent_anchor' if owner in original.pools['recent'] else 'dormant_recovery'
        elif (len(path) >= actor.config.birth_observations
              and float(window.confidences[list(path)].max()) >= actor.config.new_track_threshold):
            identity, reason = cursor, 'new'; cursor += 1
        else:
            suppressed.append(int(window.detection_keys[node])); continue
        owner_by_path[row] = identity
        actions.append(PlannedDetection(node=node, frame=original.commit_frame,
            detection_key=int(window.detection_keys[node]), identity=identity, reason=reason,
            path_keys=tuple(int(window.detection_keys[i]) for i in path),
            bbox=window.bboxes[node].copy(), confidence=float(window.confidences[node]),
            appearance=window.appearance[node].copy()))
    plan = replace(original, partition=partition, actions=tuple(actions), suppressed_keys=tuple(suppressed),
        next_identity_after=cursor,
        provisional_plan=actor._suffix_plan(window, graph, partition, original.commit_frame, owner_by_path))
    actor._validate_plan(plan, window)
    return plan


def choose_complete_cover(covers, delta, augmentation=None, cover_augmentation=None):
    """Exact joint choice within the fixed full-cover domain and shared OLD capacity.

    V(P,z)=native_optimum(P)+S3(P,z)-max_z S3(P,z)+delta(P,z).
    The per-cover offset preserves the true S3 zero control without a discontinuous
    zero-head switch. Augmentation is reserved for a separately reviewed loss.
    """
    delta = np.asarray(delta, np.float64)
    if not np.isfinite(delta).all() or delta.ndim != 1 or not covers:
        raise ValueError('invalid complete-cover correction/domain')
    best = None
    for index, cover in enumerate(covers):
        scores = cover.s3_scores.copy()
        old = cover.pair_indices >= 0
        if old.any():
            scores[old] += delta[cover.pair_indices[old]]
        if augmentation is not None:
            extra = np.asarray(augmentation[index], np.float64)
            if extra.shape != scores.shape or not np.isfinite(extra).all():
                raise ValueError('invalid complete-cover score augmentation')
            scores += extra
        variables = full_assignment(cover.sample, scores)
        value = cover.native_score + float(scores[variables].sum()) - cover.frozen_owner_optimum
        if cover_augmentation is not None:
            value += float(cover_augmentation[index])
        if best is None or value > best[3] + 1e-9:
            best = index, variables, scores, value
    return best


class ConditionalFullCoverTracker(JointOwnerDecisionTracker):
    def __init__(self, config, scorer, store, head, correction_head, device='cpu', proposal_config=None):
        super().__init__(config, scorer, store, head, device)
        if config.enable_complete_four_frame_low_only_birth:
            raise ValueError('this candidate does not change the frozen birth contract')
        self.correction_head = correction_head
        self.proposal_config = proposal_config or CoverProposalConfig()

    def plan_window(self, window, graph, gains, commit_frame=None):
        baseline = super().plan_window(window, graph, gains, commit_frame)
        native = self.last_native_plan
        proposals = complete_cover_proposals(graph, gains, self.last_anchors,
                                             native.partition, self.proposal_config)
        covers, pair_paths, pair_owners, unique_pairs = [], [], [], {}
        for index, (partition, native_score) in enumerate(proposals):
            if index == 0:
                template, sample, scores = baseline, self.last_sample, self.last_scores.copy()
                base_choice = self.last_variables.copy()
            else:
                template = template_for_cover(self, window, graph, native, partition)
                sample = sample_from_plan(self, window, graph, gains, template)
                sample['features'] = extract_features(sample, self.store, self.track_ledger,
                    self.config.short_gap, self.config.new_track_threshold)
                if self.head is None:
                    scores = sample['base'].copy()
                else:
                    with torch.inference_mode():
                        scores = self.head(torch.tensor(sample['features'], device=self.device),
                            torch.tensor(sample['base'], device=self.device),
                            torch.tensor(sample['current'], device=self.device)).cpu().numpy()
                base_choice = full_assignment(sample, scores)
            native_choice = full_assignment(sample, sample['base'])
            reconstructed_native_score = float(np.asarray(gains)[partition.selected_detection].sum()
                                                + sample['base'][native_choice].sum())
            if abs(reconstructed_native_score - native_score) > 1e-9:
                raise AssertionError('proposal native prior disagrees with complete owner LAP')
            pair_indices = np.full(len(scores), -1, np.int64)
            for v in np.flatnonzero(sample['kinds'] == 0):
                path = tuple(sample['paths'][int(sample['rows'][v])])
                owner = int(sample['identities'][v])
                pair = owner, path
                if pair not in unique_pairs:
                    unique_pairs[pair] = len(pair_paths)
                    pair_paths.append(path); pair_owners.append(owner)
                pair_indices[v] = unique_pairs[pair]
            covers.append(ConditionalCover(template, sample, scores, base_choice,
                                           native_score, pair_indices))
        packet = conditional_path_evidence(self, self.store, baseline.commit_frame, pair_paths, pair_owners)
        if self.correction_head is None:
            delta = np.zeros(len(pair_paths), np.float64)
        else:
            with torch.inference_mode():
                delta = self.correction_head(torch.tensor(packet.tokens, device=self.device),
                    torch.tensor(packet.valid, device=self.device)).cpu().numpy().astype(np.float64)
        index, variables, scores, value = choose_complete_cover(covers, delta)
        selected = covers[index]
        if index == 0 and np.array_equal(variables, self.last_variables):
            result = baseline
        else:
            result = materialize_owner_plan(self, window, graph, selected.template, selected.sample, variables)
        self.last_baseline_plan = baseline
        self.last_covers, self.last_evidence, self.last_pair_delta = tuple(covers), packet, delta
        self.last_cover_index, self.last_cover_value = index, value
        self.last_sample, self.last_variables, self.last_scores = selected.sample, variables, scores
        return result
