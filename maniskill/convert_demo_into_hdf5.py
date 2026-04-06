#!/usr/bin/env python3
"""Convert ActionBench ManiSkill demonstrations into the HDF5 format
expected by UniSkill-Policy / robomimic training.

Input layout (ActionBench sample_data):
    <data_dir>/
        0000/
            <timestamp>.h5              # obs: sensor_data/base_camera/rgb, agent/qpos, extra/tcp_pose
            <timestamp>.json            # env metadata
            <timestamp>.state.*.h5      # flat obs + actions (pd_ee_delta_pose, 7-dim)
        0001/
            ...

Output:
    <output_path>.hdf5 with structure:
        data/
            demo_0/
                obs/
                    agentview_rgb    (T, 128, 128, 3) uint8
                    eye_in_hand_rgb  (T, 128, 128, 3) uint8
                    ee_pos           (T, 3) float64
                    ee_ori           (T, 3) float64  (axis-angle)
                    joint_states     (T, 7) float64
                    gripper_states   (T, 2) float64
                actions              (T, 7) float32
                dones                (T,) uint8
                rewards              (T,) uint8
            demo_1/
                ...

Usage:
    python maniskill/convert_demo_into_hdf5.py \
        --data-dir /path/to/sample_data \
        --output datasets/action_bench/action_bench_demo.hdf5 \
        --camera-key base_camera
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def quat_wxyz_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to axis-angle (3D).

    Matches robosuite.utils.transform_utils.quat2axisangle used in LIBERO.
    """
    w = np.clip(quat_wxyz[..., 0], -1.0, 1.0)
    xyz = quat_wxyz[..., 1:]
    den = np.sqrt(1.0 - w * w)
    angle = 2.0 * np.arccos(np.abs(w))

    # Avoid division by zero for near-identity rotations
    safe = den > 1e-8
    axis_angle = np.zeros_like(xyz)
    if xyz.ndim == 1:
        if safe:
            axis_angle = xyz / den * angle * np.sign(w)
    else:
        axis_angle[safe] = (
            xyz[safe]
            / den[safe, None]
            * angle[safe, None]
            * np.sign(w[safe, None])
        )
    return axis_angle


def discover_demos(data_dir: str) -> list[dict]:
    """Find all demo directories and pair their H5/JSON files."""
    demos = []
    data_path = Path(data_dir)

    for demo_dir in sorted(data_path.iterdir()):
        if not demo_dir.is_dir():
            continue

        # Main H5 (obs with images)
        main_h5_files = [
            f for f in demo_dir.glob("*.h5") if ".state." not in f.name
        ]
        if not main_h5_files:
            continue

        # State H5 (flat obs + actions in pd_ee_delta_pose)
        state_h5_files = list(demo_dir.glob("*.state.pd_ee_delta_pose.*.h5"))

        # JSON metadata
        json_files = [
            f for f in demo_dir.glob("*.json") if ".state." not in f.name
        ]

        demos.append({
            "name": demo_dir.name,
            "main_h5": str(main_h5_files[0]),
            "state_h5": str(state_h5_files[0]) if state_h5_files else None,
            "json": str(json_files[0]) if json_files else None,
        })

    return demos


def extract_obs_from_demo(
    main_h5_path: str,
    state_h5_path: str | None,
    camera_key: str,
    img_size: int,
) -> dict:
    """Extract observations and actions from a demo's H5 files.

    Returns dict with keys matching the LIBERO HDF5 format.
    """
    with h5py.File(main_h5_path, "r") as f:
        traj = f["traj_0"]
        obs = traj["obs"]

        # RGB images
        rgb_key = f"sensor_data/{camera_key}/rgb"
        rgb_frames = obs[rgb_key][()]  # (T, H, W, 3)

        # Resize if needed
        if rgb_frames.shape[1] != img_size or rgb_frames.shape[2] != img_size:
            from PIL import Image
            resized = []
            for frame in rgb_frames:
                img = Image.fromarray(frame).resize(
                    (img_size, img_size), Image.BILINEAR
                )
                resized.append(np.array(img))
            rgb_frames = np.stack(resized)

        # Proprioceptive state
        qpos = obs["agent/qpos"][()]  # (T, 9) — 7 joints + 2 gripper
        tcp_pose = obs["extra/tcp_pose"][()]  # (T, 7) — [x,y,z, qw,qx,qy,qz]

        T_obs = qpos.shape[0]

    # Actions from state H5 (pd_ee_delta_pose, 7-dim)
    if state_h5_path is not None:
        with h5py.File(state_h5_path, "r") as f:
            actions = f["traj_0/actions"][()]  # (T-1, 7)
    else:
        # Fallback: zero actions
        actions = np.zeros((T_obs - 1, 7), dtype=np.float32)

    T = actions.shape[0]  # T_obs - 1 typically

    # Trim obs to match actions length
    rgb_frames = rgb_frames[:T]
    qpos = qpos[:T]
    tcp_pose = tcp_pose[:T]

    # Decompose into LIBERO-style obs keys
    ee_pos = tcp_pose[:, :3].astype(np.float64)               # (T, 3)
    ee_ori = quat_wxyz_to_axis_angle(tcp_pose[:, 3:7]).astype(np.float64)  # (T, 3)
    joint_states = qpos[:, :7].astype(np.float64)             # (T, 7)
    gripper_states = qpos[:, 7:9].astype(np.float64)          # (T, 2)

    return {
        "agentview_rgb": rgb_frames,        # (T, H, W, 3) uint8
        "eye_in_hand_rgb": rgb_frames,      # Use same camera as placeholder
        "ee_pos": ee_pos,                   # (T, 3) float64
        "ee_ori": ee_ori,                   # (T, 3) float64
        "joint_states": joint_states,       # (T, 7) float64
        "gripper_states": gripper_states,   # (T, 2) float64
        "actions": actions.astype(np.float64),  # (T, 7) float64
        "num_samples": T,
    }


def build_env_args(json_path: str | None) -> str:
    """Build env_args JSON string for the HDF5 data attributes."""
    meta = {}
    if json_path:
        with open(json_path) as f:
            meta = json.load(f)

    env_info = meta.get("env_info", {})
    env_id = env_info.get("env_id", "ActionBench-v1")
    env_kwargs = env_info.get("env_kwargs", {})

    env_args = {
        "type": 1,
        "env_name": env_id,
        "env_kwargs": {
            "robots": [env_kwargs.get("robot_uids", "panda")],
            "controller_configs": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2,
            },
        },
    }
    return json.dumps(env_args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert ActionBench ManiSkill demos to UniSkill-Policy HDF5"
    )
    p.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory of ActionBench demo data.",
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="Output HDF5 file path.",
    )
    p.add_argument(
        "--camera-key", type=str, default="base_camera",
        help="Camera key in the ManiSkill H5 obs (default: base_camera).",
    )
    p.add_argument(
        "--img-size", type=int, default=128,
        help="Target image size for RGB observations (default: 128).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    demos = discover_demos(args.data_dir)
    if not demos:
        print(f"No demos found in {args.data_dir}")
        return

    print(f"Found {len(demos)} demo(s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with h5py.File(args.output, "w") as out:
        data_grp = out.create_group("data")

        total_samples = 0

        for i, demo_info in enumerate(demos):
            demo_key = f"demo_{i}"
            print(f"  [{i}] {demo_info['name']} -> {demo_key}")

            obs_data = extract_obs_from_demo(
                demo_info["main_h5"],
                demo_info["state_h5"],
                args.camera_key,
                args.img_size,
            )

            T = obs_data["num_samples"]
            total_samples += T

            demo_grp = data_grp.create_group(demo_key)
            demo_grp.attrs["num_samples"] = T

            # Observations
            obs_grp = demo_grp.create_group("obs")
            for key in ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori",
                        "joint_states", "gripper_states"):
                obs_grp.create_dataset(key, data=obs_data[key])

            # Actions
            demo_grp.create_dataset("actions", data=obs_data["actions"])

            # Dones and rewards (all zeros, last step done)
            dones = np.zeros(T, dtype=np.uint8)
            dones[-1] = 1
            demo_grp.create_dataset("dones", data=dones)
            demo_grp.create_dataset("rewards", data=dones.copy())

        # Data group attributes
        env_args_str = build_env_args(demos[0]["json"] if demos else None)
        data_grp.attrs["env_args"] = env_args_str
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["num_demos"] = len(demos)

    print(f"\nWrote {args.output}")
    print(f"  Demos: {len(demos)}")
    print(f"  Total samples: {total_samples}")


if __name__ == "__main__":
    main()
