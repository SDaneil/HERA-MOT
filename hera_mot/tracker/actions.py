"""Label-free history features and joint owner actuation on fixed native DD."""
from dataclasses import replace
import numpy as np
import torch

from ..models.ias import full_assignment
from .lifecycle import AnchoredPartition, PlannedDetection
from .allocation import UnifiedBoundaryFourFrameTracker


FEATURE_NAMES = (
    'OLD', 'NEW', 'HOLD', 'current', 'native_gain',
    'root_confidence', 'path_min_confidence', 'path_mean_confidence', 'path_max_confidence',
    'path_observations', 'path_span', 'root_width', 'root_height', 'root_center_x', 'root_center_y',
    'path_cosine_min', 'path_cosine_mean', 'path_motion_mean', 'path_motion_max', 'path_size_change',
    'best_DD_out', 'positive_DD_out', 'birth_eligible',
    'gap', 'recent', 'dormant', 'log_writes', 'log_age',
    'last_dx', 'last_dy', 'predicted_dx', 'predicted_dy', 'log_width_ratio', 'log_height_ratio', 'last_iou',
    'cosine_last', 'cosine_birth', 'cosine_last4_mean', 'cosine_last8_mean', 'cosine_sampled_history_mean',
    'cosine_last8_min', 'cosine_last8_max', 'cosine_last8_std', 'history_appearance_drift',
    'history_confidence_mean', 'history_confidence_min', 'history_high_confidence_fraction',
    'history_gap_mean', 'history_gap_std', 'velocity_x', 'velocity_y', 'history_size_drift',
    'row_old_count', 'row_best_gain', 'row_second_gain', 'gain_minus_row_best',
    'row_recent_best', 'row_dormant_best', 'owner_competitors', 'gain_minus_owner_other_best',
)


class BlindObservationStore:
    """Explicit whitelist: GT fields cannot be accessed by feature extraction."""
    def __init__(self, data):
        if 'labels' in data or 'gt_ids' in data:
            raise ValueError('feature input contains GT')
        self.keys = np.asarray(data['keys'], np.int64)
        self.frames = np.asarray(data['frames'], np.int64)
        self.boxes = np.asarray(data['boxes'], np.float32)
        self.scores = np.asarray(data['scores'], np.float32)
        self.appearance = np.asarray(data['graph_appearance'], np.float32)
        self.width, self.height = int(data['width']), int(data['height'])
        self.index = {int(k): i for i, k in enumerate(self.keys)}

    def indices(self, keys):
        return np.asarray([self.index[int(k)] for k in keys], np.int64)


def box_iou(a, b):
    intersection = np.maximum(0., np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / max(1e-8, a[2:].prod() + b[2:].prod() - intersection))


def extract_features(sample, store, ledger, short_gap=3, birth_threshold=.6246494191492591):
    """Reads only <=t-1 committed history and <=t+3 window observations."""
    count = len(sample['base'])
    result = np.zeros((count, len(FEATURE_NAMES)), np.float32)
    frame = sample['frame']
    histories = {}
    for identity in set(int(i) for i in sample['identities'] if i >= 0):
        keys = list(ledger[identity])
        selected = sorted(set(list(range(min(4, len(keys)))) +
                              np.linspace(0, len(keys) - 1, min(32, len(keys))).astype(int).tolist() +
                              list(range(max(0, len(keys) - 8), len(keys)))))
        indices = store.indices([keys[i] for i in selected])
        if np.any(store.frames[indices] >= frame):
            raise AssertionError('uncommitted history accessed')
        last = store.indices(keys[-8:])
        birth = store.index[int(keys[0])]
        histories[identity] = (keys, indices, last, birth)
    outgoing = sample.get('DD_out', {})
    for row, path in enumerate(sample['paths']):
        node_indices = store.indices(path)
        if store.frames[node_indices].min() < frame or store.frames[node_indices].max() > frame + 3:
            raise AssertionError('four-frame window horizon exceeded')
        current = int(store.frames[node_indices[0]]) == frame
        positions = np.flatnonzero(sample['rows'] == row)
        # Provisional scores are native constants and do not need features.
        if not current:
            continue
        root = node_indices[0]
        root_box = store.boxes[root].astype(np.float64)
        root_feature = store.appearance[root]
        height = max(1., root_box[3])
        boxes = store.boxes[node_indices].astype(np.float64)
        center = boxes[:, :2] + boxes[:, 2:] / 2.
        motion = np.linalg.norm(np.diff(center, axis=0), axis=1) / height
        scores = store.scores[node_indices]
        path_cosine = store.appearance[node_indices] @ root_feature
        dd = outgoing.get(int(path[0]), (0., 0.))
        shared = [float(store.scores[root]), float(scores.min()), float(scores.mean()), float(scores.max()),
            len(path) / 4., (store.frames[node_indices[-1]] - store.frames[root]) / 3.,
            root_box[2] / store.width, root_box[3] / store.height,
            (root_box[0] + root_box[2] / 2) / store.width, (root_box[1] + root_box[3] / 2) / store.height,
            float(path_cosine.min()), float(path_cosine.mean()),
            float(motion.mean()) if len(motion) else 0., float(motion.max()) if len(motion) else 0.,
            float(np.abs(np.log(np.maximum(boxes[-1, 2:], 1e-4) / np.maximum(boxes[0, 2:], 1e-4))).mean()),
            dd[0], dd[1] / 16., float(sample['eligible'][row])]
        old_positions = positions[sample['kinds'][positions] == 0]
        gains = sample['base'][old_positions]
        ranked = sorted(gains.tolist(), reverse=True)
        recent_gains, dormant_gains = [], []
        for pos in old_positions:
            identity = int(sample['identities'][pos])
            last = histories[identity][2][-1]
            (recent_gains if frame - store.frames[last] <= short_gap else dormant_gains).append(sample['base'][pos])
        row_context = [len(gains) / 16., ranked[0] if ranked else 0., ranked[1] if len(ranked) > 1 else 0.]
        for pos in positions:
            kind = int(sample['kinds'][pos])
            gain = float(sample['base'][pos])
            old_features = [0.] * 29
            owner_context = [0., 0.]
            if kind == 0:
                identity = int(sample['identities'][pos])
                keys, selected, recent, birth = histories[identity]
                last = recent[-1]
                old_box = store.boxes[last].astype(np.float64)
                old_center = old_box[:2] + old_box[2:] / 2.
                gap = int(frame - store.frames[last])
                velocity = np.zeros(2)
                if len(recent) > 1:
                    prev = recent[-2]
                    prev_center = store.boxes[prev, :2] + store.boxes[prev, 2:] / 2.
                    velocity = (old_center - prev_center) / max(1, int(store.frames[last] - store.frames[prev]))
                difference = (center[0] - old_center) / height
                prediction_error = (center[0] - old_center - velocity * gap) / height
                cosines = store.appearance[recent] @ root_feature
                hist_gaps = np.diff(store.frames[recent])
                sampled_mean = store.appearance[selected].mean(0)
                norm = max(1e-8, float(np.linalg.norm(sampled_mean)))
                drift = 1. - float(store.appearance[birth] @ store.appearance[last])
                old_features = [gap / 40., float(gap <= short_gap), float(gap > short_gap),
                    np.log1p(len(keys)) / 8., np.log1p(frame - store.frames[birth]) / 8.,
                    *difference, *prediction_error,
                    *np.log(np.maximum(root_box[2:], 1e-4) / np.maximum(old_box[2:], 1e-4)), box_iou(root_box, old_box),
                    float(cosines[-1]), float(store.appearance[birth] @ root_feature),
                    float(cosines[-4:].mean()), float(cosines.mean()), float(sampled_mean @ root_feature) / norm,
                    float(cosines.min()), float(cosines.max()), float(cosines.std()), drift,
                    float(store.scores[recent].mean()), float(store.scores[recent].min()),
                    float((store.scores[recent] >= birth_threshold).mean()),
                    float(hist_gaps.mean()) / 40. if len(hist_gaps) else 0.,
                    float(hist_gaps.std()) / 40. if len(hist_gaps) else 0.,
                    *(velocity / height),
                    float(np.abs(np.log(np.maximum(old_box[2:], 1e-4) / np.maximum(store.boxes[birth, 2:], 1e-4))).mean())]
                other = np.flatnonzero((sample['identities'] == identity) & (sample['rows'] != row))
                owner_context = [len(other) / 16., gain - float(sample['base'][other].max(initial=0.))]
            value = [float(kind == 0), float(kind == 1), float(kind == 2), 1., gain,
                     *shared, *old_features, *row_context, gain - (ranked[0] if ranked else 0.),
                     max(recent_gains, default=0.), max(dormant_gains, default=0.), *owner_context]
            if len(value) != len(FEATURE_NAMES):
                raise AssertionError(f'feature schema length {len(value)} != {len(FEATURE_NAMES)}')
            result[pos] = np.clip(value, -8., 8.)
    if not np.isfinite(result).all():
        raise AssertionError('nonfinite deployable features')
    return result


def sample_from_trace(log, frame):
    """Whitelist geometric/native fields; never reads trace GT or primary."""
    paths = [p['uids'] for p in log['paths']]
    rows, cols, kinds, owners, base = [], [], [], [], []
    reference, eligible, old = [], [], None
    for row in log['owner_plan']['rows']:
        r = int(row['path'])
        eligible.append(bool(row['NEW_eligible']))
        reference.append(len(rows) + next(i for i, o in enumerate(row['options']) if o['column'] == row['reference']))
        for option in row['options']:
            kind = ('OLD', 'NEW', 'HOLD').index(option['action'])
            rows.append(r); cols.append(int(option['column'])); kinds.append(kind)
            owners.append(int(option['identity']) if kind == 0 else -1)
            base.append(float(option['native']))
            if kind == 2:
                value = int(option['column']) - len(paths) - r
                if old is not None and old != value:
                    raise AssertionError('inconsistent private columns')
                old = value
    dd = {}
    for e in log['selected_detection_edges']:
        source = int(e['source'])
        best, positive = dd.get(source, (0., 0))
        dd[source] = (max(best, e['gain']), positive + int(e['gain'] > 0))
    return dict(frame=int(frame), paths=paths, path_count=len(paths), old_count=old or 0,
                rows=np.asarray(rows, np.int64), columns=np.asarray(cols, np.int64), kinds=np.asarray(kinds, np.int8),
                identities=np.asarray(owners, np.int64), base=np.asarray(base, np.float64),
                reference=np.asarray(reference, np.int64), eligible=eligible, DD_out=dd)


def sample_from_plan(tracker, window, graph, gains, plan):
    # Describe only the selected physical DD paths, matching frozen trace
    # materialization. No future/GT labels are present in this interface.
    anchors, partition = tracker.last_anchors, plan.partition
    paths = [[int(window.detection_keys[n]) for n in p] for p in partition.paths]
    count, old = len(paths), len(anchors.anchor_ids)
    action_nodes = {a.node for a in plan.actions}
    rows = []
    for row, path in enumerate(partition.paths):
        node = path[0]
        current = int(window.frames[node]) == plan.commit_frame
        birth = current and len(path) >= tracker.config.birth_observations and float(
            window.confidences[list(path)].max()) >= tracker.config.new_track_threshold
        options = []
        for (index, target), gain in zip(anchors.edges, anchors.gains):
            if int(target) == node and gain > 0:
                options.append(dict(column=int(index), action='OLD', identity=int(anchors.anchor_ids[index]), native=float(gain)))
        if birth:
            options.append(dict(column=old + row, action='NEW', identity=None, native=0.))
        options.append(dict(column=old + count + row, action='HOLD', identity=None, native=0.))
        owner = partition.anchor_by_path[row]
        reference = (anchors.anchor_ids.index(owner) if owner is not None else
                     old + row if node in action_nodes else old + count + row)
        rows.append(dict(path=row, NEW_eligible=bool(birth), reference=reference, options=options))
    log = dict(paths=[dict(uids=p) for p in paths], owner_plan=dict(rows=rows),
               selected_detection_edges=[dict(source=int(window.detection_keys[s]), target=int(window.detection_keys[t]), gain=float(g))
                   for (s, t), g, chosen in zip(graph.edges, gains, partition.selected_detection) if chosen])
    sample = sample_from_trace(log, plan.commit_frame)
    sample['current'] = np.asarray([int(window.frames[partition.paths[r][0]]) == plan.commit_frame for r in sample['rows']], np.float64)
    return sample


def materialize_owner_plan(tracker, window, graph, original, sample, variables):
    columns = sample['columns'][variables]
    owners = tuple(int(sample['identities'][v]) if sample['kinds'][v] == 0 else None for v in variables)
    anchors = tracker.last_anchors
    selected = np.zeros_like(original.partition.selected_anchor)
    edge_lookup = {(int(i), int(n)): e for e, (i, n) in enumerate(anchors.edges)}
    for row, owner in enumerate(owners):
        if owner is not None:
            selected[edge_lookup[anchors.anchor_ids.index(owner), original.partition.paths[row][0]]] = True
    partition = AnchoredPartition(original.partition.paths, owners, original.partition.selected_detection.copy(), selected)
    mapping = {n: r for r, path in enumerate(partition.paths) for n in path}
    actions, suppressed, cursor = [], [], tracker.next_identity
    owner_by_path = {r: owner for r, owner in enumerate(owners) if owner is not None}
    for node in np.flatnonzero(window.frames == original.commit_frame).tolist():
        row = mapping[node]
        kind = int(sample['kinds'][variables[row]])
        if kind == 2:
            suppressed.append(int(window.detection_keys[node])); continue
        if kind == 0:
            identity = owners[row]
            reason = 'recent_anchor' if identity in original.pools['recent'] else 'dormant_recovery'
        else:
            if not sample['eligible'][row]:
                raise AssertionError('birth eligibility bypassed')
            identity, reason = cursor, 'new'; cursor += 1
        owner_by_path[row] = identity
        actions.append(PlannedDetection(node=node, frame=original.commit_frame,
            detection_key=int(window.detection_keys[node]), identity=identity, reason=reason,
            path_keys=tuple(int(window.detection_keys[n]) for n in partition.paths[row]),
            bbox=window.bboxes[node].copy(), confidence=float(window.confidences[node]), appearance=window.appearance[node].copy()))
    plan = replace(original, partition=partition, actions=tuple(actions), suppressed_keys=tuple(suppressed),
        next_identity_after=cursor, provisional_plan=tracker._suffix_plan(window, graph, partition, original.commit_frame, owner_by_path))
    tracker._validate_plan(plan, window)
    return plan


class JointOwnerDecisionTracker(UnifiedBoundaryFourFrameTracker):
    def __init__(self, config, scorer, store, head=None, device='cpu'):
        super().__init__(config, scorer)
        self.store, self.head, self.device = store, head, torch.device(device)

    def plan_window(self, window, graph, gains, commit_frame=None):
        native = super().plan_window(window, graph, gains, commit_frame)
        sample = sample_from_plan(self, window, graph, gains, native)
        features = extract_features(sample, self.store, self.track_ledger, self.config.short_gap, self.config.new_track_threshold)
        sample['features'] = features
        if self.head is None:
            scores = sample['base'].copy()
        else:
            with torch.inference_mode():
                scores = self.head(torch.as_tensor(features, device=self.device),
                    torch.as_tensor(sample['base'], device=self.device), torch.as_tensor(sample['current'], device=self.device)).cpu().numpy()
        variables = full_assignment(sample, scores)
        plan = materialize_owner_plan(self, window, graph, native, sample, variables)
        self.last_native_plan, self.last_sample = native, sample
        self.last_variables, self.last_scores = variables, scores
        if np.array_equal(variables, sample['reference']):
            return native
        return plan
