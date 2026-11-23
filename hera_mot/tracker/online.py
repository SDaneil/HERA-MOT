"""Score association actions and commit the selected assignments."""
from types import SimpleNamespace

import numpy as np
import torch

from ..models.ias import full_assignment
from ..models.history_graph import history_graph_input, history_pair_indices
from . import actions as joint
from . import cover as complete
from . import hes as pixel_runtime


class MinimalCoverTracker(complete.ConditionalFullCoverTracker):
    """Refine native association scores with history and appearance evidence."""
    def __init__(self, config, scorer, store, model, *, device='cpu', proposal_config=None,
                 pixel_appearance=None, pixel_valid=None):
        spec = model.spec
        if proposal_config is None:
            proposal_config = complete.CoverProposalConfig(max_covers=spec.max_covers, max_cut_edges=16)
        super().__init__(config, scorer, store, getattr(model, 's3', None), None,
                         device=device, proposal_config=proposal_config)
        self.model, self.spec = model, spec
        self.pixel_calls = self.history_calls = 0
        self.last_history_delta = np.empty(0, np.float32)
        if spec.pixel:
            if pixel_appearance is None or pixel_valid is None:
                raise ValueError('HES requires secondary descriptors and a validity mask')
            if pixel_appearance.shape != (len(store.keys), 3584) or pixel_valid.shape != (len(store.keys),):
                raise ValueError('Secondary descriptors and validity mask must align with detections')
            self.pixel_appearance, self.pixel_valid = pixel_appearance, pixel_valid
            self.pixel_smooth, self.pixel_last_frame = {}, {}
        elif pixel_appearance is not None or pixel_valid is not None:
            raise ValueError('Disable secondary inputs when HES is disabled')

    def commit_plan(self, plan, window):
        result = super().commit_plan(plan, window)
        if self.spec.pixel:
            for action in result.committed:
                assert action.frame == plan.commit_frame
                index = self.store.index[int(action.detection_key)]
                assert self.pixel_valid[index]
                owner = int(action.identity)
                assert self.pixel_last_frame.get(owner, -1) < action.frame
                self.pixel_smooth[owner] = pixel_runtime.smooth_update(self.pixel_smooth.get(owner),
                    self.pixel_appearance[index], self.store.scores[index], self.scorer.config)
                self.pixel_last_frame[owner] = action.frame
            live = set(self.states)
            self.pixel_smooth = {k: v for k, v in self.pixel_smooth.items() if k in live}
            self.pixel_last_frame = {k: v for k, v in self.pixel_last_frame.items() if k in live}
        return result

    def _pair_correction(self, window, graph, frame, paths, owners):
        # History consumes pair addresses, never pixel tokens. NoPixel therefore
        # avoids both the secondary descriptor and private pixel/native evidence.
        if self.spec.pixel:
            packet = pixel_runtime.pixel_path_evidence(self, self.store, frame, paths, owners)
            self.pixel_calls += 1
            ordinary = self.model.pixel(torch.tensor(packet.tokens, device=self.device),
                                        torch.tensor(packet.valid, device=self.device))
        else:
            packet = SimpleNamespace(paths=tuple(tuple(p) for p in paths), owners=tuple(owners), decision_frame=frame)
            ordinary = None
        self.last_history_delta = np.empty(0, np.float32)
        if self.spec.history:
            h = history_graph_input(self, window, graph, self.last_anchors)
            ix = history_pair_indices(packet, window, h)
            self.history_calls += 1
            delta = self.model.history_delta(window, graph, h)
            self.last_history_delta = delta.detach().cpu().numpy()
            addition = torch.cat((delta, delta.new_zeros(1)))[torch.as_tensor(ix, device=self.device)]
            combined = addition if ordinary is None else ordinary + addition
        else:
            combined = ordinary
        result = np.zeros(len(paths), np.float64) if combined is None else combined.cpu().numpy().astype(np.float64)
        if result.shape != (len(paths),) or not np.isfinite(result).all():
            raise ValueError('Expected one finite score correction per candidate pair')
        return packet, result

    def plan_window(self, window, graph, gains, commit_frame=None):
        # Refer to original MODULE attributes so process-local certified fast
        # bindings remain live. Do not capture pre-install aliases.
        baseline = joint.JointOwnerDecisionTracker.plan_window(self, window, graph, gains, commit_frame)
        native = self.last_native_plan
        proposals = complete.complete_cover_proposals(graph, gains, self.last_anchors,
                                                      native.partition, self.proposal_config)
        covers, pair_paths, pair_owners, unique_pairs = [], [], [], {}
        for index, (partition, native_score) in enumerate(proposals):
            if index == 0:
                template, sample, scores = baseline, self.last_sample, self.last_scores.copy()
                base_choice = self.last_variables.copy()
            else:
                template = complete.template_for_cover(self, window, graph, native, partition)
                sample = complete.sample_from_plan(self, window, graph, gains, template)
                sample['features'] = complete.extract_features(sample, self.store, self.track_ledger,
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
            reconstructed = float(np.asarray(gains)[partition.selected_detection].sum() + sample['base'][native_choice].sum())
            if abs(reconstructed - native_score) > 1e-9:
                raise AssertionError('Native proposal score disagrees with the assignment score')
            pair_indices = np.full(len(scores), -1, np.int64)
            for v in np.flatnonzero(sample['kinds'] == 0):
                path = tuple(sample['paths'][int(sample['rows'][v])])
                owner = int(sample['identities'][v]); pair = owner, path
                if pair not in unique_pairs:
                    unique_pairs[pair] = len(pair_paths)
                    pair_paths.append(path); pair_owners.append(owner)
                pair_indices[v] = unique_pairs[pair]
            covers.append(complete.ConditionalCover(template, sample, scores, base_choice, native_score, pair_indices))
        with torch.inference_mode():
            packet, delta = self._pair_correction(window, graph, baseline.commit_frame, pair_paths, pair_owners)
        index, variables, scores, value = complete.choose_complete_cover(covers, delta)
        selected = covers[index]
        if index == 0 and np.array_equal(variables, self.last_variables):
            result = baseline
        else:
            result = complete.materialize_owner_plan(self, window, graph, selected.template, selected.sample, variables)
        self.last_baseline_plan = baseline
        self.last_covers, self.last_evidence, self.last_pair_delta = tuple(covers), packet, delta
        self.last_cover_index, self.last_cover_value = index, value
        self.last_sample, self.last_variables, self.last_scores = selected.sample, variables, scores
        return result
