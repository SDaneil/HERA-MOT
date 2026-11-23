"""Original history-independent detection window container."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionWindow:
    """GT-free observations for one causal window.

    ``bboxes`` are TLWH observations after any causal GMC transformation that
    the experiment elects to use.  Raw writer boxes must remain separate.
    """

    detection_keys: np.ndarray
    frames: np.ndarray
    bboxes: np.ndarray
    confidences: np.ndarray
    appearance: np.ndarray
    start_frame: int
    end_frame: int
    image_width: int
    image_height: int

    def __post_init__(self):
        keys = np.asarray(self.detection_keys)
        frames = np.asarray(self.frames)
        bboxes = np.asarray(self.bboxes)
        confidence = np.asarray(self.confidences)
        appearance = np.asarray(self.appearance)
        count = len(keys)
        if frames.shape != (count,) or confidence.shape != (count,):
            raise ValueError("detection keys, frames and confidences must align")
        if bboxes.shape != (count, 4) or appearance.ndim != 2 or len(appearance) != count:
            raise ValueError("bboxes/appearance must have shapes [N,4]/[N,D]")
        if len(np.unique(keys)) != count:
            raise ValueError("DetKey values must be unique inside a window")
        if self.start_frame > self.end_frame:
            raise ValueError("invalid causal window bounds")
        if count and ((frames < self.start_frame).any() or (frames > self.end_frame).any()):
            raise ValueError("detection frame lies outside the causal window")
        if count > 1:
            expected = np.lexsort((keys.astype(np.int64), frames.astype(np.int64)))
            if not np.array_equal(expected, np.arange(count)):
                raise ValueError("window detections must be sorted by (frame, DetKey)")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if not np.isfinite(bboxes).all() or not np.isfinite(appearance).all():
            raise ValueError("window features must be finite")
        if count and (bboxes[:, 2:] <= 0).any():
            raise ValueError("detection boxes must have positive width and height")
        if not np.isfinite(confidence).all():
            raise ValueError("detection confidences must be finite")

    @property
    def window_size(self) -> int:
        return int(self.end_frame - self.start_frame + 1)
