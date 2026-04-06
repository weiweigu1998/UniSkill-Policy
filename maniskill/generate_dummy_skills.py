#!/usr/bin/env python3
"""Generate dummy skill embeddings for testing UniSkill-Policy training.

Creates random skill .npy files matching the demo structure in an HDF5 file,
so that robomimic training can run without real skill extraction.

Usage:
    python maniskill/generate_dummy_skills.py \
        --hdf5 datasets/action_bench/action_bench_demo.hdf5 \
        --skill-dir skills \
        --skill-dim 64 \
        --aug-num 5
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Generate dummy skill embeddings")
    p.add_argument("--hdf5", type=str, required=True, help="Input HDF5 dataset path")
    p.add_argument("--skill-dir", type=str, default="skills", help="Output skill directory")
    p.add_argument("--skill-dim", type=int, default=64, help="Skill embedding dimension")
    p.add_argument("--aug-num", type=int, default=5, help="Number of augmented skill variants")
    args = p.parse_args()

    task_name = os.path.basename(os.path.splitext(args.hdf5)[0])

    with h5py.File(args.hdf5, "r") as f:
        demo_keys = sorted(f["data"].keys())
        for demo_id in demo_keys:
            num_samples = f["data"][demo_id].attrs["num_samples"]

            out_dir = os.path.join(args.skill_dir, task_name, demo_id)
            os.makedirs(out_dir, exist_ok=True)

            # Base skill: random unit-norm embeddings
            # Shape (T, 1, skill_dim) — the extra dim is expected by
            # diffusion_policy.py which indexes skill[:,0,:].
            skill = np.random.randn(num_samples, 1, args.skill_dim).astype(np.float32)
            skill /= np.linalg.norm(skill, axis=-1, keepdims=True) + 1e-8
            np.save(os.path.join(out_dir, "base.npy"), skill)

            # Augmented variants
            for aug_idx in range(args.aug_num):
                aug_skill = skill + 0.1 * np.random.randn(num_samples, 1, args.skill_dim).astype(np.float32)
                aug_skill /= np.linalg.norm(aug_skill, axis=-1, keepdims=True) + 1e-8
                np.save(os.path.join(out_dir, f"aug_{aug_idx}.npy"), aug_skill)

            print(f"  {demo_id}: {num_samples} samples -> {out_dir}")

    print(f"Done. Skills written to {args.skill_dir}/{task_name}/")


if __name__ == "__main__":
    main()
