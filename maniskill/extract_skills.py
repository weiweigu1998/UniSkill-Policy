#!/usr/bin/env python3
"""Extract skill embeddings from postprocessed robot trajectories using a trained IDM.

Reads the per-sample pkl dataset produced by
``scripts/process_training_trajectories.py``:

    <samples-root>/data/<idx>.pkl          # one pkl per (task, demo, timestep)
    <samples-root>/meta/index.jsonl        # {"idx","task","demo_id","t"} per line

Sample idxs are grouped by ``(task, demo_id)`` from ``index.jsonl`` and sorted
by ``t`` to reconstruct each demo's frame sequence. For each demo we run the
UniSkill IDM on consecutive frame pairs to produce per-timestep skill vectors,
saved in the layout robomimic training expects:

    <skill_dir>/<task>/demo_<i>/base.npy   (T, 1, skill_dim)
    <skill_dir>/<task>/demo_<i>/aug_0.npy  ... aug_<N-1>.npy

The demo index ``i`` is the position of ``demo_id`` in the sorted demo-id list
within each task — matching ``convert_lfo_to_robomimic``'s ``demo_<i>``
numbering (both sort demo ids the same way).

Preprocessing notes (kept consistent with how the IDM was *trained* in
``UniSkill/diffusion/dataset/lfo_benchmark_dataset.py``):
  * Depth-model input: frames are scaled to [0, 1], bicubically resized to the
    depth model's native 518x518, then ImageNet-normalized. All on-GPU.
  * IDM visual input: frames scaled to [0, 1] (the IDM applies its own
    ImageNet Normalize inside ``forward_encoder``), resized to ``idm_resolution``.

Both the depth preprocessing and the IDM/depth models run on the CUDA device;
the previous h5-based implementation did the depth resize/normalize on CPU via
the HuggingFace image processor, which dominated wall time (~25 s/batch vs
~0.9 s of GPU compute).

Usage:
    python maniskill/extract_skills.py \
        --samples-root /root/data/lfo_benchmark/postprocessed_robot_trajectories \
        --idm-checkpoint /path/to/idm.pth \
        --skill-dir /path/to/skills \
        --aug-num 5 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

# Add UniSkill root to path for IDM import
_UNISKILL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "UniSkill")
)
if os.path.isdir(_UNISKILL_ROOT) and _UNISKILL_ROOT not in sys.path:
    sys.path.insert(0, _UNISKILL_ROOT)

from dynamics.idm import IDM

# ImageNet statistics — the normalization Depth-Anything-V2 expects, and the
# same stats the IDM training pipeline's depth_processor applied.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DEPTH_INPUT_SIZE = 518  # Depth-Anything-V2 native resolution.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract skills from postprocessed robot trajectories")
    p.add_argument("--samples-root", type=str, required=True,
                   help="postprocessed_robot_trajectories root holding data/ and meta/.")
    p.add_argument("--task", type=str, default=None,
                   help="Optional: only extract skills for this single task name.")
    p.add_argument("--camera-key", type=str, default="base_camera",
                   help="Which observation/<camera> key to read from each pkl (default: base_camera).")
    p.add_argument("--idm-checkpoint", type=str, required=True,
                   help="Path to pretrained IDM weights (.pth).")
    p.add_argument("--skill-dir", type=str, required=True,
                   help="Output directory for skill .npy files.")
    p.add_argument("--depth-model", type=str,
                   default="depth-anything/Depth-Anything-V2-Small-hf",
                   help="HuggingFace depth estimator model name.")
    p.add_argument("--aug-num", type=int, default=5,
                   help="Number of augmented skill variants to generate.")
    p.add_argument("--aug-noise", type=float, default=0.1,
                   help="Noise scale for augmented skills.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size (frame pairs) for IDM inference.")
    p.add_argument("--skill-interval", type=int, default=1,
                   help="Temporal offset between frame pairs for IDM.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract demos even if their skill files already exist.")
    # IDM architecture params
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--skill-dim", type=int, default=64)
    p.add_argument("--out-dim", type=int, default=768)
    p.add_argument("--idm-resolution", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_idm(args: argparse.Namespace, device: torch.device) -> IDM:
    """Load pretrained IDM weights and place the model on ``device``."""
    idm = IDM(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        skill_dim=args.skill_dim,
        out_dim=args.out_dim,
        idm_resolution=args.idm_resolution,
    )
    checkpoint = torch.load(args.idm_checkpoint, map_location="cpu", weights_only=False)
    idm.load_state_dict(checkpoint)
    idm.requires_grad_(False)
    idm.eval()
    idm.to(device)
    # Verify the whole module actually landed on the CUDA device.
    param_dev = next(idm.parameters()).device
    if param_dev.type != device.type:
        raise RuntimeError(f"IDM failed to move to {device}; params are on {param_dev}")
    print(f"  IDM on {param_dev}")
    return idm


def get_predicted_depth(output):
    """Extract depth prediction from model output."""
    if hasattr(output, "predicted_depth"):
        return output.predicted_depth
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unexpected depth estimator output type: {type(output)}")


def discover_demos(samples_root: str, task_filter: str | None
                   ) -> list[tuple[str, list[tuple[str, list[int]]]]]:
    """Group sample idxs by ``(task, demo_id)`` from ``meta/index.jsonl``.

    Returns ``[(task, [(demo_id, [sorted sample_idx, ...]), ...]), ...]`` with
    demo ids sorted so ``demo_<i>`` lines up with ``convert_lfo_to_robomimic``.
    """
    index_path = os.path.join(samples_root, "meta", "index.jsonl")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"Expected {index_path}. Run scripts/process_training_trajectories.py first."
        )
    groups: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            groups[e["task"]][e["demo_id"]].append((int(e["t"]), int(e["idx"])))

    tasks: list[tuple[str, list[tuple[str, list[int]]]]] = []
    for task in sorted(groups):
        if task_filter is not None and task != task_filter:
            continue
        demos = []
        for demo_id in sorted(groups[task]):
            # Sort by t so the idx list is chronological (t=0, t=1, ...).
            pairs = sorted(groups[task][demo_id])
            demos.append((demo_id, [idx for _t, idx in pairs]))
        if demos:
            tasks.append((task, demos))
    return tasks


def load_demo_frames(samples_root: str, sample_idxs: list[int], camera_key: str) -> np.ndarray:
    """Stack ``observation/<camera_key>`` across a demo's sample pkls → (T,H,W,3) uint8."""
    data_dir = os.path.join(samples_root, "data")
    frames = []
    obs_key = f"observation/{camera_key}"
    for idx in sample_idxs:
        with open(os.path.join(data_dir, f"{idx}.pkl"), "rb") as fh:
            sample = pickle.load(fh)
        frames.append(sample[obs_key])
    return np.stack(frames, axis=0)


def _depth_pixel_values(frames_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 (B,H,W,3) → Depth-Anything pixel_values (B,3,518,518), all on GPU.

    Matches the IDM training pipeline: scale to [0,1], bicubic-resize to the
    depth model's native resolution, ImageNet-normalize.
    """
    x = torch.from_numpy(frames_uint8).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(_DEPTH_INPUT_SIZE, _DEPTH_INPUT_SIZE),
                      mode="bicubic", align_corners=False).clamp_(0.0, 1.0)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def _visual_tensor(frames_uint8: np.ndarray, device: torch.device, resolution: int) -> torch.Tensor:
    """uint8 (B,H,W,3) → IDM visual input (B,3,res,res) in [0,1], on GPU.

    The IDM applies its own ImageNet Normalize inside ``forward_encoder``, so we
    only scale + resize here.
    """
    x = torch.from_numpy(frames_uint8).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    return F.interpolate(x, size=(resolution, resolution),
                         mode="bilinear", align_corners=False)


def extract_skills_from_frames(
    frames: np.ndarray,
    idm: IDM,
    depth_estimator,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    """Extract skill embeddings from RGB frames using the IDM.

    Args:
        frames: (T, H, W, 3) uint8 RGB frames.

    Returns:
        skills: (T, 1, skill_dim) float32 skill embeddings.
    """
    T = frames.shape[0]
    interval = args.skill_interval
    num_pairs = T - interval
    if num_pairs <= 0:
        return np.zeros((T, 1, args.skill_dim), dtype=np.float32)

    resolution = args.idm_resolution
    all_skills = []

    for start in range(0, num_pairs, args.batch_size):
        end = min(start + args.batch_size, num_pairs)
        curr = frames[start:end]
        nxt = frames[start + interval:end + interval]

        with torch.no_grad():
            # Visual pair for the IDM (B,2,3,res,res), [0,1].
            visual_pair = torch.stack(
                [_visual_tensor(curr, device, resolution),
                 _visual_tensor(nxt, device, resolution)], dim=1)

            # Depth: one batched forward over curr+next, on GPU.
            depth_in = torch.cat([_depth_pixel_values(curr, device),
                                  _depth_pixel_values(nxt, device)], dim=0)
            depth_out = get_predicted_depth(depth_estimator(pixel_values=depth_in))
            if depth_out.ndim == 4 and depth_out.size(1) == 1:
                depth_out = depth_out.squeeze(1)
            curr_depth, next_depth = torch.chunk(depth_out, 2, dim=0)
            depth_pair = torch.stack([curr_depth, next_depth], dim=1)
            depth_pair = F.interpolate(depth_pair, size=(resolution, resolution),
                                       mode="bilinear", align_corners=False)

            skills = idm(depth_pair, visual_pair, return_skill=True)

        all_skills.append(skills.float().cpu().numpy())

    skills_array = np.concatenate(all_skills, axis=0)

    # Pad to length T by repeating the last skill for the final `interval` frames.
    if skills_array.shape[0] < T:
        pad_count = T - skills_array.shape[0]
        padding = np.tile(skills_array[-1:], (pad_count,) + (1,) * (skills_array.ndim - 1))
        skills_array = np.concatenate([skills_array, padding], axis=0)

    return skills_array.astype(np.float32)


def _demo_already_done(out_dir: str, aug_num: int) -> bool:
    """True if base.npy and every aug_<i>.npy already exist for this demo."""
    if not os.path.isfile(os.path.join(out_dir, "base.npy")):
        return False
    return all(os.path.isfile(os.path.join(out_dir, f"aug_{i}.npy"))
               for i in range(aug_num))


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        print(f"WARNING: --device={args.device} is not a CUDA device; extraction will be slow.")

    tasks = discover_demos(args.samples_root, args.task)
    if not tasks:
        print(f"No demos found under {args.samples_root}")
        return
    total_demos = sum(len(demos) for _t, demos in tasks)
    print(f"Discovered {total_demos} demos across {len(tasks)} task(s) in {args.samples_root}")

    print(f"Loading IDM from {args.idm_checkpoint}...")
    idm = load_idm(args, device)

    print(f"Loading depth estimator: {args.depth_model}...")
    depth_estimator = AutoModelForDepthEstimation.from_pretrained(args.depth_model)
    depth_estimator.requires_grad_(False)
    depth_estimator.eval()
    depth_estimator.to(device)
    print(f"  depth estimator on {next(depth_estimator.parameters()).device}")

    done = skipped = 0
    for task_name, demos in tasks:
        print(f"\nTask {task_name!r}: {len(demos)} demos")
        for demo_idx, (demo_id, sample_idxs) in enumerate(demos):
            out_dir = os.path.join(args.skill_dir, task_name, f"demo_{demo_idx}")
            if not args.overwrite and _demo_already_done(out_dir, args.aug_num):
                skipped += 1
                continue

            t0 = time.time()
            frames = load_demo_frames(args.samples_root, sample_idxs, args.camera_key)
            skills = extract_skills_from_frames(frames, idm, depth_estimator, device, args)

            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "base.npy"), skills)
            for aug_idx in range(args.aug_num):
                aug = skills + args.aug_noise * np.random.randn(*skills.shape).astype(np.float32)
                np.save(os.path.join(out_dir, f"aug_{aug_idx}.npy"), aug)
            done += 1
            print(f"  demo_{demo_idx} (id {demo_id}): {frames.shape[0]} frames "
                  f"-> {skills.shape}  [{time.time() - t0:.1f}s]")

    print(f"\nDone. Extracted {done} demos, skipped {skipped} already-present. "
          f"Skills under {args.skill_dir}/")


if __name__ == "__main__":
    main()
