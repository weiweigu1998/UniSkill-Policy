#!/usr/bin/env python3
"""Extract skill embeddings from ManiSkill demonstrations using a trained IDM.

Reads the converted LIBERO-format HDF5 (from convert_demo_into_hdf5.py),
runs the UniSkill IDM on consecutive frame pairs to produce per-timestep
skill vectors, and saves them in the layout expected by robomimic training:

    <skill_dir>/<task_name>/<demo_id>/base.npy     (T, 1, skill_dim)

For skill augmentation, multiple noisy variants are also saved:

    <skill_dir>/<task_name>/<demo_id>/aug_0.npy
    <skill_dir>/<task_name>/<demo_id>/aug_1.npy
    ...

Usage:
    python maniskill/extract_skills.py \
        --hdf5 /workspace/data/maniskill_hdf5/action_bench_demo.hdf5 \
        --idm-checkpoint /workspace/checkpoints/uniskill/UniSkill_final_weight/idm.pth \
        --skill-dir /workspace/data/maniskill_hdf5/skills \
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
    p = argparse.ArgumentParser(description="Extract skills from ManiSkill HDF5 demos")
    p.add_argument("--hdf5", type=str, required=True,
                    help="Path to the converted LIBERO-format HDF5 file.")
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


def main():
    args = parse_args()
    device = torch.device(args.device)

    task_name = os.path.basename(os.path.splitext(args.hdf5)[0])

    print(f"Loading IDM from {args.idm_checkpoint}...")
    idm = load_idm(args, device)

    print(f"Loading depth estimator: {args.depth_model}...")
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model)
    depth_estimator = AutoModelForDepthEstimation.from_pretrained(args.depth_model)
    depth_estimator.requires_grad_(False)
    depth_estimator.eval()
    depth_estimator.to(device)

    print(f"Processing {args.hdf5}...")
    with h5py.File(args.hdf5, "r") as f:
        demo_keys = sorted(f["data"].keys())

        for demo_id in demo_keys:
            demo_grp = f["data"][demo_id]

            # Load RGB frames from agentview
            if "obs/agentview_rgb" in demo_grp:
                frames = demo_grp["obs/agentview_rgb"][()]
            else:
                print(f"  {demo_id}: no agentview_rgb found, skipping")
                continue

            print(f"  {demo_id}: {frames.shape[0]} frames, extracting skills...")

            skills = extract_skills_from_frames(
                frames, idm, depth_processor, depth_estimator, device, args
            )

            # Save base skill
            out_dir = os.path.join(args.skill_dir, task_name, demo_id)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "base.npy"), skills)

            # Save augmented variants
            for aug_idx in range(args.aug_num):
                aug_skills = skills + args.aug_noise * np.random.randn(
                    *skills.shape
                ).astype(np.float32)
                np.save(os.path.join(out_dir, f"aug_{aug_idx}.npy"), aug_skills)

            print(f"    -> {out_dir} ({skills.shape})")

    print(f"\nDone. Skills saved to {args.skill_dir}/{task_name}/")


if __name__ == "__main__":
    main()
