# Tracker inputs

Run `python -m hera_mot` once per video. The tracker reads cached detector and appearance-encoder outputs and writes one MOTChallenge-format result file.

## Observations

Pass one `.npz` archive with `--input`. The detection arrays share row order. `N` is the number of detections across the entire input sequence.

| Field | Shape | Contents |
| --- | --- | --- |
| `keys` | `(N,)` | Unique nonnegative integer detection IDs |
| `frames` | `(N,)` | One-based integer frame numbers within the input sequence or clip |
| `boxes` | `(N, 4)` | Zero-based left, top, width, and height in source-image pixels. Width and height must be positive. |
| `scores` | `(N,)` | Detection confidence in `[0, 1]` |
| `graph_appearance` | `(N, 3584)` | L2-normalized primary appearance descriptors |
| `secondary_appearance` | `(N, 3584)` | L2-normalized auxiliary appearance descriptors |
| `width`, `height`, `length` | `()` | Positive integer scalars for image dimensions and total frame count, including frames with no detections |

You can store the descriptors in separate `.npy` files with `--primary` and `--secondary`. These files use the same row order as the archive and take precedence over the corresponding embedded descriptors.

Sampling starts at local frame 1. For MOT17 sparse validation, clip each FRCNN training sequence to original frames `floor(L / 2) + 1` through `L`, where `L` is that sequence's frame count. Rebase the first frame of the clip to 1. The tracker keeps empty selected frames and processes detections in frame and detection-key order.

## Camera transforms

Pass a second `.npz` archive with `--warps`.

| Field | Shape | Contents |
| --- | --- | --- |
| `warps` | `(K, 2, 3)` or `(K, 3, 3)` | Floating-point transforms from the previous selected image to the current one |
| `source_pairs` | `(K, 2)` | Integer pairs of local source-frame numbers in sampling order |

`K` counts all selected frames, including empty ones. All transforms must be finite and nonsingular. The first transform is identity for the pair `(1, 1)`. At stride 2 the next pairs are `(1, 3), (3, 5), ...`. Estimate each warp between those selected images. Pass `--identity-warps` to disable camera-motion compensation.

## Checkpoint and output

Pass the Full association model's tensor state with `--checkpoint` or place it at the relative path in the selected profile. A weights-only-compatible PyTorch file can hold the state dictionary directly or under `model` or `state_dict`.

`--output` writes comma-separated `frame,id,left,top,width,height,confidence,-1,-1,-1` rows. The tracker adds one to the left and top coordinates. Source-frame numbers are used by default. Use `--frame-offset` to add back leading frames from a clipped sequence, or `--frame-numbering sampled` to write consecutive sampled indices.
