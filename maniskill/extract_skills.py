#!/usr/bin/env python3
"""Extract robot-trajectory skill embeddings using the upstream UniSkill IDM.

Thin wrapper around ``UniSkill/extract_skill.py`` — the robot-side skill
extraction shares one core implementation with the human-side
``scripts/extract_uniskill_skill_lfo.py`` rather than duplicating the
IDM + Depth-Anything forward loop.

This script only adds the LfO-specific glue:

  * **Discovery** — reads ``postprocessed_robot_trajectories/meta/index.jsonl``
    and groups sample idxs by ``(task, demo_id)``.
  * **Frame loading** — stacks ``observation/<camera>`` from each demo's
    per-sample pkls into a ``(T, H, W, 3)`` array.
  * **Output formatting** — pads the upstream ``(T-interval, skill_dim)``
    result to ``(T, 1, skill_dim)`` and writes the robomimic layout that
    ``SequenceDataset(goal_mode="skill")`` consumes:

        <skill-dir>/<task>/demo_<i>/base.npy        (T, 1, skill_dim)
        <skill-dir>/<task>/demo_<i>/aug_<k>.npy

The core frames→IDM→skills computation is entirely
``extract_skill.extract_latents_for_demo`` — unmodified upstream code.

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
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoImageProcessor

# Resolve ``from extract_skill import ...`` to the upstream UniSkill module.
_UNISKILL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "UniSkill")
)
if os.path.isdir(_UNISKILL_ROOT) and _UNISKILL_ROOT not in sys.path:
    sys.path.insert(0, _UNISKILL_ROOT)

from extract_skill import extract_latents_for_demo, load_depth_estimator, load_idm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract robot skills via the upstream UniSkill IDM")
    p.add_argument("--samples-root", required=True,
                   help="postprocessed_robot_trajectories root holding data/ and meta/.")
    p.add_argument("--task", default=None, help="Optional: only this task.")
    p.add_argument("--camera-key", default="base_camera",
                   help="Which observation/<camera> key to read from each pkl.")
    p.add_argument("--idm-checkpoint", required=True, help="Pretrained IDM weights (.pth).")
    p.add_argument("--skill-dir", required=True, help="Output directory for skill .npy files.")
    p.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--aug-num", type=int, default=5,
                   help="Number of noisy augmented skill variants per demo.")
    p.add_argument("--aug-noise", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skill-interval", type=int, default=1,
                   help="Temporal offset between IDM frame pairs.")
    p.add_argument("--prefetch-workers", type=int, default=2,
                   help="Background threads for batch prep (passed to extract_latents_for_demo).")
    p.add_argument("--amp", action="store_true", help="Enable autocast in the model forwards.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract demos even if their skill files already exist.")
    # IDM architecture params — consumed by the upstream load_idm.
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--skill-dim", type=int, default=64)
    p.add_argument("--out-dim", type=int, default=768)
    p.add_argument("--idm-resolution", type=int, default=224)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def discover_demos(samples_root: str, task_filter: str | None
                   ) -> list[tuple[str, list[tuple[str, list[int]]]]]:
    """Group sample idxs by ``(task, demo_id)`` from ``meta/index.jsonl``.

    Returns ``[(task, [(demo_id, [sorted-by-t sample_idx, ...]), ...]), ...]``
    with demo ids sorted so ``demo_<i>`` lines up with
    ``convert_lfo_to_robomimic``'s per-task HDF5 ordering.
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
        demos = [(demo_id, [idx for _t, idx in sorted(groups[task][demo_id])])
                 for demo_id in sorted(groups[task])]
        if demos:
            tasks.append((task, demos))
    return tasks


def load_demo_frames(samples_root: str, sample_idxs: list[int], camera_key: str) -> np.ndarray:
    """Stack ``observation/<camera_key>`` across a demo's pkls → (T,H,W,3) uint8."""
    data_dir = os.path.join(samples_root, "data")
    obs_key = f"observation/{camera_key}"
    frames = []
    for idx in sample_idxs:
        with open(os.path.join(data_dir, f"{idx}.pkl"), "rb") as fh:
            frames.append(pickle.load(fh)[obs_key])
    return np.stack(frames, axis=0)


def _demo_already_done(out_dir: str, aug_num: int) -> bool:
    if not os.path.isfile(os.path.join(out_dir, "base.npy")):
        return False
    return all(os.path.isfile(os.path.join(out_dir, f"aug_{k}.npy"))
               for k in range(aug_num))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    tasks = discover_demos(args.samples_root, args.task)
    if not tasks:
        print(f"No demos found under {args.samples_root}")
        return
    total = sum(len(demos) for _t, demos in tasks)
    print(f"Discovered {total} demos across {len(tasks)} task(s) in {args.samples_root}")

    print(f"Loading IDM from {args.idm_checkpoint}...")
    idm, skill_dim = load_idm(args, device)
    print(f"Loading depth estimator: {args.depth_model}...")
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model)
    depth_estimator = load_depth_estimator(args.depth_model, device)

    done = skipped = 0
    for task_name, demos in tasks:
        print(f"\nTask {task_name!r}: {len(demos)} demos")
        for demo_idx, (demo_id, sample_idxs) in enumerate(demos):
            out_dir = os.path.join(args.skill_dir, task_name, f"demo_{demo_idx}")
            if not args.overwrite and _demo_already_done(out_dir, args.aug_num):
                skipped += 1
                continue

            frames = load_demo_frames(args.samples_root, sample_idxs, args.camera_key)
            T = frames.shape[0]

            # Core extraction: unmodified upstream UniSkill code.
            skills_2d = extract_latents_for_demo(
                frames,
                idm=idm,
                idm_resolution=args.idm_resolution,
                depth_processor=depth_processor,
                depth_estimator=depth_estimator,
                device=device,
                batch_size=args.batch_size,
                interval=args.skill_interval,
                skill_dim=skill_dim,
                prefetch_workers=args.prefetch_workers,
                use_amp=args.amp,
            )  # (T - interval, skill_dim)

            # Pad to T (repeat the last skill) and add the singleton sequence
            # dim that robomimic's goal_mode="skill" loader indexes as [:, 0, :].
            if skills_2d.shape[0] == 0:
                skills_2d = np.zeros((T, skill_dim), dtype=np.float32)
            elif skills_2d.shape[0] < T:
                pad = np.tile(skills_2d[-1:], (T - skills_2d.shape[0], 1))
                skills_2d = np.concatenate([skills_2d, pad], axis=0)
            skills = skills_2d[:, None, :].astype(np.float32)  # (T, 1, skill_dim)

            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "base.npy"), skills)
            for k in range(args.aug_num):
                aug = skills + args.aug_noise * np.random.randn(*skills.shape).astype(np.float32)
                np.save(os.path.join(out_dir, f"aug_{k}.npy"), aug)
            done += 1
            print(f"  demo_{demo_idx} (id {demo_id}): {T} frames -> {skills.shape}")

    print(f"\nDone. Extracted {done} demos, skipped {skipped} already-present. "
          f"Skills under {args.skill_dir}/")


if __name__ == "__main__":
    main()
