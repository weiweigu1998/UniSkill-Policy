#!/usr/bin/env python3
"""Extract skill embeddings from ManiSkill demonstrations using a trained IDM.

Walks the LfO sample-data layout directly:

    <data_dir>/training_trajectories/<task>/<traj_id>/<timestamp>.h5

For each robot trajectory we run the UniSkill IDM on consecutive frame pairs
to produce per-timestep skill vectors, and save them in the layout expected by
robomimic training:

    <skill_dir>/<task>/demo_<i>/base.npy     (T, 1, skill_dim)

The demo index ``i`` is assigned by sorted ``traj_id`` order within each task,
matching how :mod:`convert_demo_into_hdf5` packs trajectories into per-task
HDF5s. Robomimic's dataset loader keys on ``task_name = basename(hdf5)`` and
demo_id from the HDF5, so as long as both scripts walk the trajectories in the
same order the skill files line up.

For skill augmentation, multiple noisy variants are also saved:

    <skill_dir>/<task>/demo_<i>/aug_0.npy
    <skill_dir>/<task>/demo_<i>/aug_1.npy
    ...

Usage:
    python maniskill/extract_skills.py \
        --data-dir /path/to/sample_data \
        --idm-checkpoint /path/to/idm.pth \
        --skill-dir /path/to/skills \
        --aug-num 5
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Add UniSkill root to path for IDM import
_UNISKILL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "UniSkill")
)
if os.path.isdir(_UNISKILL_ROOT) and _UNISKILL_ROOT not in sys.path:
    sys.path.insert(0, _UNISKILL_ROOT)

from dynamics.idm import IDM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract skills from ManiSkill LfO demos")
    p.add_argument("--data-dir", type=str, required=True,
                    help="LfO sample-data root containing training_trajectories/<task>/<traj_id>/...")
    p.add_argument("--task", type=str, default=None,
                    help="Optional: only extract skills for this single task name.")
    p.add_argument("--camera-key", type=str, default="base_camera",
                    help="Camera under traj_0/obs/sensor_data/<camera>/rgb (default: base_camera).")
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
    p.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for IDM inference.")
    p.add_argument("--skill-interval", type=int, default=1,
                    help="Temporal offset between frame pairs for IDM.")
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
    """Load pretrained IDM model."""
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
    return idm


def get_predicted_depth(output):
    """Extract depth prediction from model output."""
    if hasattr(output, "predicted_depth"):
        return output.predicted_depth
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unexpected depth estimator output type: {type(output)}")


def extract_skills_from_frames(
    frames: np.ndarray,
    idm: IDM,
    depth_processor,
    depth_estimator,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    """Extract skill embeddings from RGB frames using the IDM.

    Args:
        frames: (T, H, W, 3) uint8 RGB frames

    Returns:
        skills: (T, 1, skill_dim) float32 skill embeddings
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
        curr_indices = np.arange(start, end)
        next_indices = curr_indices + interval

        # Prepare visual tensors: (B, 3, H, W) normalized to [0, 1]
        curr_tensor = (
            torch.from_numpy(frames[curr_indices])
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
        next_tensor = (
            torch.from_numpy(frames[next_indices])
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )

        # Resize to IDM resolution
        visual_curr = F.interpolate(
            curr_tensor.to(device),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )
        visual_next = F.interpolate(
            next_tensor.to(device),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )
        visual_pair = torch.stack([visual_curr, visual_next], dim=1)

        # Compute depth features
        depth_inputs = (
            [frames[i] for i in curr_indices] +
            [frames[i] for i in next_indices]
        )
        depth_batch = depth_processor(
            images=depth_inputs,
            do_rescale=False,
            return_tensors="pt",
        )
        depth_batch = {k: v.to(device) for k, v in depth_batch.items()}

        with torch.no_grad():
            depth_outputs = depth_estimator(**depth_batch)
            depth_outputs = get_predicted_depth(depth_outputs)
            if depth_outputs.ndim == 4 and depth_outputs.size(1) == 1:
                depth_outputs = depth_outputs.squeeze(1)

            curr_depth, next_depth = torch.chunk(depth_outputs, 2, dim=0)
            depth_pair = torch.stack([curr_depth, next_depth], dim=1)
            depth_pair = F.interpolate(
                depth_pair,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )

            skills = idm(depth_pair, visual_pair, return_skill=True)

        all_skills.append(skills.float().cpu().numpy())

    # Concatenate all batches: (num_pairs, 1, skill_dim)
    skills_array = np.concatenate(all_skills, axis=0)

    # Pad to length T by repeating the last skill for the final `interval` frames
    if skills_array.shape[0] < T:
        pad_count = T - skills_array.shape[0]
        padding = np.tile(skills_array[-1:], (pad_count,) + (1,) * (skills_array.ndim - 1))
        skills_array = np.concatenate([skills_array, padding], axis=0)

    # IDM returns (B, 1, skill_dim) -- already has the sequence dimension
    # expected by diffusion_policy.py's batch["skill"][:,0,:] indexing.
    return skills_array.astype(np.float32)


def discover_tasks(training_root: str, task_filter: str | None) -> list[tuple[str, list[str]]]:
    """Return ``[(task_name, [traj_h5_path, ...]), ...]`` for the LfO layout.

    Trajectories within each task are sorted by ``traj_id`` (the demo subfolder
    name) so the skill index lines up with ``convert_demo_into_hdf5``'s
    ``demo_<i>`` numbering inside the per-task HDF5.
    """
    tasks: list[tuple[str, list[str]]] = []
    if not os.path.isdir(training_root):
        return tasks
    for task_entry in sorted(os.listdir(training_root)):
        if task_filter is not None and task_entry != task_filter:
            continue
        task_dir = os.path.join(training_root, task_entry)
        if not os.path.isdir(task_dir):
            continue
        h5_paths: list[str] = []
        for traj_entry in sorted(os.listdir(task_dir)):
            traj_dir = os.path.join(task_dir, traj_entry)
            if not os.path.isdir(traj_dir):
                continue
            candidates = [
                os.path.join(traj_dir, f)
                for f in sorted(os.listdir(traj_dir))
                if f.endswith(".h5") and ".state." not in f
            ]
            if candidates:
                h5_paths.append(candidates[0])
        if h5_paths:
            tasks.append((task_entry, h5_paths))
    return tasks


def main():
    args = parse_args()
    device = torch.device(args.device)

    training_root = os.path.join(args.data_dir, "training_trajectories")
    if not os.path.isdir(training_root):
        # Tolerate flat layouts (older sample data) for backwards compat.
        training_root = args.data_dir

    tasks = discover_tasks(training_root, args.task)
    if not tasks:
        print(f"No tasks found under {training_root}")
        return

    print(f"Loading IDM from {args.idm_checkpoint}...")
    idm = load_idm(args, device)

    print(f"Loading depth estimator: {args.depth_model}...")
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model)
    depth_estimator = AutoModelForDepthEstimation.from_pretrained(args.depth_model)
    depth_estimator.requires_grad_(False)
    depth_estimator.eval()
    depth_estimator.to(device)

    rgb_key = f"traj_0/obs/sensor_data/{args.camera_key}/rgb"
    for task_name, h5_paths in tasks:
        print(f"\nTask {task_name!r}: {len(h5_paths)} trajector{'y' if len(h5_paths) == 1 else 'ies'}")

        for demo_idx, h5_path in enumerate(h5_paths):
            demo_id = f"demo_{demo_idx}"

            with h5py.File(h5_path, "r") as f:
                if rgb_key not in f:
                    print(f"  {demo_id}: missing {rgb_key} in {h5_path}, skipping")
                    continue
                frames = f[rgb_key][()]

            print(f"  {demo_id}: {frames.shape[0]} frames ({os.path.basename(h5_path)})")

            skills = extract_skills_from_frames(
                frames, idm, depth_processor, depth_estimator, device, args,
            )

            out_dir = os.path.join(args.skill_dir, task_name, demo_id)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "base.npy"), skills)
            for aug_idx in range(args.aug_num):
                aug_skills = skills + args.aug_noise * np.random.randn(
                    *skills.shape
                ).astype(np.float32)
                np.save(os.path.join(out_dir, f"aug_{aug_idx}.npy"), aug_skills)
            print(f"    -> {out_dir} ({skills.shape})")

    print(f"\nDone. Skills saved under {args.skill_dir}/")


if __name__ == "__main__":
    main()
