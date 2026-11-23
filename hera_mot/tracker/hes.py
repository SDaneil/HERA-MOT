"""A separately trained image descriptor supplements, never replaces, native evidence."""
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from .evidence import conditional_path_evidence, TOKEN_NAMES


PIXEL_TOKEN_NAMES = ('pixel_committed_smooth', 'pixel_hypothetical_smooth',
    'pixel_last', 'pixel_birth', 'pixel_gallery_mean', 'pixel_gallery_max',
    'pixel_gallery_min', 'pixel_path_prefix_mean', 'pixel_path_prefix_min')
COMBINED_TOKEN_NAMES = TOKEN_NAMES + PIXEL_TOKEN_NAMES


def smooth_update(previous, appearance, confidence, config):
    if previous is None:
        return appearance.copy()
    trust = (float(confidence) - config.trust_floor) / (1 - config.trust_floor)
    alpha = config.appearance_alpha + (1 - config.appearance_alpha) * (1 - trust)
    value = alpha * previous + (1 - alpha) * appearance
    return value / max(float(np.linalg.norm(value)), 1e-12)


def pixel_path_evidence(tracker, store, decision_frame, paths, owners):
    old = conditional_path_evidence(tracker, store, decision_frame, paths, owners)
    extra = np.zeros((len(paths), 4, len(PIXEL_TOKEN_NAMES)), np.float32)
    prepared = {}
    for row, (path, owner) in enumerate(zip(old.paths, old.owners)):
        history = tracker.track_ledger[owner]
        hi = store.indices(history); pi = store.indices(path)
        assert tracker.pixel_valid[hi].all() and tracker.pixel_valid[pi].all()
        assert (store.frames[hi] < decision_frame).all()
        assert tracker.pixel_last_frame[owner] == int(store.frames[hi[-1]])
        if owner not in prepared:
            selected = sorted(set(range(min(4, len(history)))) |
                set(np.linspace(0, len(history) - 1, min(32, len(history))).astype(int).tolist()) |
                set(range(max(0, len(history) - 8), len(history))))
            prepared[owner] = tracker.pixel_appearance[hi[selected]], tracker.pixel_smooth[owner].copy()
        gallery, smooth = prepared[owner]; smooth = smooth.copy()
        committed = smooth.copy(); last, birth = tracker.pixel_appearance[hi[-1]], tracker.pixel_appearance[hi[0]]
        prefix = []
        for step, index in enumerate(pi):
            appearance = tracker.pixel_appearance[index]
            history_cosines = gallery @ appearance
            prefix_cosines = np.asarray(prefix) @ appearance if prefix else np.array([0.])
            extra[row, step] = [committed @ appearance, smooth @ appearance, last @ appearance,
                birth @ appearance, history_cosines.mean(), history_cosines.max(), history_cosines.min(),
                prefix_cosines.mean(), prefix_cosines.min()]
            smooth = smooth_update(smooth, appearance, store.scores[index], tracker.scorer.config)
            prefix.append(appearance)
    tokens = np.concatenate((old.tokens, extra), axis=2)
    assert np.isfinite(tokens).all(); tokens.setflags(write=False)
    return replace(old, tokens=tokens)


class PixelPathCorrection(nn.Module):
    def __init__(self, correction=.5, width=64):
        super().__init__()
        assert 0 < correction <= 1 and width > 0
        self.correction = float(correction)
        self.network = nn.Sequential(nn.Linear(4 * (len(COMBINED_TOKEN_NAMES) + 1), width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)

    def forward(self, tokens, valid):
        assert tokens.shape == (len(valid), 4, len(COMBINED_TOKEN_NAMES)) and valid.dtype == torch.bool
        assert torch.isfinite(tokens).all()
        assert not len(valid) or (valid[:, 0].all() and not (valid[:, 1:] & ~valid[:, :-1]).any())
        x = torch.cat((tokens.masked_fill(~valid[..., None], 0.), valid[..., None].to(tokens.dtype)), dim=2)
        return self.correction * torch.tanh(self.network(x.flatten(1)).squeeze(1))
