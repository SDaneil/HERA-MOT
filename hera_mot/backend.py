"""Strictly online HERA-MOT association on cached detection observations."""

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import torch

from .models.hgrnet import HistorySourceContrastGraph
from .models.ias import JointOwnerDecisionHead
from .models.window_graph import StrongWindowGraph
from .tracker.actions import BlindObservationStore
from .tracker.boost import BoostBoundaryConfig, BoostTrackBoundaryCost
from .tracker.cover import CoverProposalConfig
from .tracker.graph import ConnectionGraph
from .tracker.hes import PixelPathCorrection
from .tracker.lifecycle import FixedLagConfig
from .tracker.observations import DetectionWindow
from .tracker.online import MinimalCoverTracker


class _FullAssociation(torch.nn.Module):
    """Full association model for frame-by-frame tracking."""

    def __init__(self):
        super().__init__()
        self.spec = SimpleNamespace(history=True, s3=True, pixel=True, max_covers=1)
        self.c4 = StrongWindowGraph(3584)
        self.c4.time = torch.nn.Embedding(8, 64)
        self.history_score = torch.nn.Linear(64, 1, bias=False)
        self.bound = .25
        self.s3 = JointOwnerDecisionHead(60, hidden=64, correction=1.)
        self.pixel = PixelPathCorrection(correction=.5, width=64)
        self.float()

    history_features = HistorySourceContrastGraph.history_features
    history_delta = HistorySourceContrastGraph.history_delta


def make_model(state):
    """Load a Full 64-tensor state dictionary, or initialize for a demo."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(497)
        model = _FullAssociation()
    if state is not None:
        expected = model.state_dict()
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError('Expected the complete Full 64-tensor association state.')
        for key, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid Full association tensor: {key}')
        model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def track(raw, secondary, warps, settings, model):
    """Yield sampled frame, identity, original TLWH, and detection confidence.

    The keys and rows in ``raw`` and ``secondary`` share the same order.
    The frame clock starts at 1 and advances across empty sampled frames.
    """
    store = BlindObservationStore(raw)
    length = int(raw['length'])
    secondary = np.asarray(secondary, dtype=np.float32)
    if store.appearance.shape != (len(store.keys), 3584) or secondary.shape != store.appearance.shape:
        raise ValueError('Primary and secondary descriptors both require shape (N, 3584).')
    if np.any(np.diff(store.frames) < 0):
        raise ValueError('Detections must be sorted by sampled frame, then key.')
    if length < 1 or np.any((store.frames < 1) | (store.frames > length)):
        raise ValueError('Detections must be on the declared one-based sampled frame clock.')
    if settings.get('birth_observations', 1) != 1:
        raise ValueError('Strict frame-by-frame tracking requires birth_observations=1.')

    config = FixedLagConfig(
        admission_threshold=settings['admission_threshold'],
        new_track_threshold=settings['new_track_threshold'],
        short_gap=settings['short_gap'],
        keep_gap=settings['keep_gap'],
        birth_observations=1,
    )
    scorer = BoostTrackBoundaryCost(
        BoostBoundaryConfig(trust_floor=settings.get('trust_floor', config.admission_threshold)),
        warps,
        gate=settings.get('gate', 'native'),
    )
    actor = MinimalCoverTracker(
        config,
        scorer,
        store,
        model,
        device=next(model.parameters()).device,
        proposal_config=CoverProposalConfig(max_covers=1, max_cut_edges=16),
        pixel_appearance=secondary,
        pixel_valid=np.ones(len(store.keys), dtype=bool),
    )
    # A one-frame window has no detection-to-detection graph edges.
    edges = np.empty((0, 2), dtype=np.int64)
    edge_features = np.empty((0, 7), dtype=np.float32)
    gains = np.empty(0, dtype=np.float64)

    for frame in range(1, length + 1):
        lo, hi = np.searchsorted(store.frames, [frame, frame + 1])
        rows = np.arange(lo, hi, dtype=np.int64)
        rows = rows[store.scores[rows] >= config.admission_threshold]
        window = DetectionWindow(
            store.keys[rows], store.frames[rows], store.boxes[rows], store.scores[rows],
            store.appearance[rows], frame, frame, store.width, store.height,
        )
        graph = ConnectionGraph(window.frames, edges, edge_features)
        with torch.inference_mode():
            plan = actor.plan_window(window, graph, gains, frame)
            result = actor.commit_plan(plan, window)
        for item in result.committed:
            yield int(item.frame), int(item.identity), *map(float, item.bbox), float(item.confidence)
