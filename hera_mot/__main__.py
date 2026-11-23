"""Run HERA-MOT on cached detections from a single video sequence."""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from pickle import UnpicklingError
import tempfile

import numpy as np

from .inputs import FEATURE_DIM, identity_warps, load_observations, load_warps, sample_observations


def load_profile(path, name, stride):
    config = json.loads(path.read_text(encoding="utf-8"))
    name = config["default_profile"] if name is None else name
    try:
        settings = config["profiles"][name]
    except KeyError as error:
        profiles = ", ".join(config["profiles"])
        raise ValueError(f"unknown profile {name!r}. Available profiles: {profiles}") from error
    stride = settings["stride"] if stride is None else stride
    if type(stride) is not int or stride < 1 or stride not in settings["allowed_strides"]:
        raise ValueError(f"stride {stride!r} is outside the settings for {config['dataset']} {name}")
    return config["dataset"], name, settings, stride


def load_checkpoint(path):
    import torch

    if not path.is_file():
        raise ValueError(f"checkpoint not found: {path}. Provide a Full association checkpoint with --checkpoint")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except UnpicklingError as error:
        raise ValueError("export the Full association tensors as a weights-only PyTorch checkpoint") from error
    if isinstance(payload, Mapping):
        for name in ("model", "state_dict"):
            if isinstance(payload.get(name), Mapping):
                payload = payload[name]
                break
    if not isinstance(payload, Mapping) or not payload or not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError("expected a tensor-only state dictionary for the Full association model")
    return dict(payload)


def demo_observations(stride):
    """One object, an empty selected step, and two competing detections."""
    primary = np.zeros((4, FEATURE_DIM), dtype=np.float32)
    primary[:, 0] = 1
    secondary = np.roll(primary, 1, axis=1)
    raw = {
        "keys": np.arange(4, dtype=np.int64),
        "frames": np.array([1, 1 + 2 * stride, 1 + 2 * stride, 1 + 3 * stride]),
        "boxes": np.array([[20, 30, 24, 50], [22, 30, 24, 50], [26, 30, 24, 50], [23, 30, 24, 50]], dtype=np.float32),
        "scores": np.array([.95, .94, .74, .91], dtype=np.float32),
        "graph_appearance": primary,
        "width": 128,
        "height": 128,
        "length": 1 + 3 * stride,
    }
    return raw, secondary


def write_mot(path, rows, source_frames, numbering, offset):
    """Write one-based MOTChallenge box origins and the chosen frame numbering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", dir=path.parent, delete=False) as handle:
            staged = Path(handle.name)
            for step, identity, x, y, width, height, confidence in rows:
                frame = step if numbering == "sampled" else int(source_frames[step - 1]) + offset
                handle.write(f"{frame},{identity},{x + 1:.3f},{y + 1:.3f},{width:.3f},{height:.3f},{confidence:.6f},-1,-1,-1\n")
                count += 1
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="dataset JSON configuration")
    parser.add_argument("--profile", help="named profile from the dataset configuration")
    parser.add_argument("--stride", type=int, help="override the configured sampling stride")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="per-sequence observation NPZ")
    inputs.add_argument("--demo", action="store_true", help="run a synthetic sequence with untrained weights")
    parser.add_argument("--primary", type=Path, help="primary embeddings NPY in the same input row order")
    parser.add_argument("--secondary", type=Path, help="auxiliary embeddings NPY in the same input row order")
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--warps", type=Path, help="selected-frame camera transforms NPZ")
    motion.add_argument("--identity-warps", action="store_true", help="run with camera-motion compensation disabled")
    parser.add_argument("--checkpoint", type=Path, help="Full association weights")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--output", type=Path, required=True, help="MOTChallenge output text file")
    parser.add_argument("--frame-numbering", choices=("source", "sampled"), default="source")
    parser.add_argument("--frame-offset", type=int, default=0, help="preceding frames when the input is a clipped segment")
    args = parser.parse_args()
    if args.frame_offset < 0 or (args.frame_numbering == "sampled" and args.frame_offset):
        parser.error("frame-offset must be nonnegative and is only used with source numbering")
    if args.demo and any((args.primary, args.secondary, args.warps, args.checkpoint)):
        parser.error("demo uses synthetic features, identity transforms, and untrained weights")

    try:
        dataset, profile, settings, stride = load_profile(args.config, args.profile, args.stride)
        if args.demo:
            raw, secondary = demo_observations(stride)
            state = None
        else:
            raw, secondary = load_observations(args.input, primary=args.primary, secondary=args.secondary)
            checkpoint = args.checkpoint or args.config.resolve().parent.parent / settings["checkpoint"]
            state = load_checkpoint(checkpoint)
        sampled, secondary, clock = sample_observations(raw, secondary, stride)
        if args.warps is not None:
            warps = load_warps(args.warps, clock)
        elif args.demo or args.identity_warps or settings["camera_motion"] == "identity":
            warps = identity_warps(len(clock))
        else:
            raise ValueError("provide --warps for ECC or explicitly select --identity-warps")

        from .backend import make_model, track

        model = make_model(state).to(args.device).eval()
        rows = track(sampled, secondary, warps, settings, model)
        count = write_mot(args.output, rows, clock, args.frame_numbering, args.frame_offset)
    except ImportError as error:
        parser.exit(2, f"hera_mot: {error}. Install the dependencies in requirements.txt.\n")
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(2, f"hera_mot: {error}\n")
    mode = " (synthetic)" if args.demo else ""
    print(f"{dataset} {profile}{mode}: {count} detections across {len(clock)} sampled steps -> {args.output}")


if __name__ == "__main__":
    main()
