#!/usr/bin/env python3
"""Convert ActionBench ManiSkill demonstrations into the HDF5 format
expected by UniSkill-Policy / robomimic training.

Input layout (LfO sample_data):
    <data_dir>/
        training_trajectories/<task>/<traj_id>/
            <timestamp>.h5              # obs: sensor_data/base_camera/rgb, agent/qpos, extra/tcp_pose
            <timestamp>.json            # env metadata
            <timestamp>.state.*.h5      # flat obs + actions (pd_ee_delta_pose, 7-dim)
        video_demonstrations/<task>/...  # NOT used here (phase 1 only).

Output: **one HDF5 per task** at ``<output_dir>/<task>.hdf5``. Robomimic's skill
loader keys on ``task_name = basename(hdf5)``, so the per-task HDF5 basename
must match the ``<task>`` folder used by ``extract_skills.py`` when writing
``<skill_dir>/<task>/demo_<i>/...``.

    <output_dir>/<task>.hdf5
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

The ``demo_<i>`` index is assigned by sorted ``traj_id`` order within the task,
matching the ordering ``extract_skills.py`` uses when writing skill .npy files.

Usage:
    python maniskill/convert_demo_into_hdf5.py \
        --data-dir /path/to/sample_data \
        --output-dir datasets/action_bench \
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


def _build_demo_record(traj_dir: Path) -> dict | None:
    """Pair the main H5, state H5, and JSON inside one trajectory folder."""
    main_h5_files = [
        f for f in traj_dir.glob("*.h5") if ".state." not in f.name
    ]
    if not main_h5_files:
        return None
    state_h5_files = list(traj_dir.glob("*.state.pd_ee_delta_pose.*.h5"))
    json_files = [
        f for f in traj_dir.glob("*.json") if ".state." not in f.name
    ]
    return {
        "name": traj_dir.name,
        "main_h5": str(main_h5_files[0]),
        "state_h5": str(state_h5_files[0]) if state_h5_files else None,
        "json": str(json_files[0]) if json_files else None,
    }


def discover_tasks(data_dir: str, task_filter: str | None = None) -> list[tuple[str, list[dict]]]:
    """Walk the LfO layout and return ``[(task_name, [demo_record, ...]), ...]``.

    Falls back to the older flat layout (``<data_dir>/<traj_id>/``) so existing
    flat-format datasets keep working. In the flat case the synthetic task name
    is the basename of ``data_dir``.
    """
    data_path = Path(data_dir)
    training_root = data_path / "training_trajectories"

    if training_root.is_dir():
        tasks: list[tuple[str, list[dict]]] = []
        for task_dir in sorted(training_root.iterdir()):
            if not task_dir.is_dir():
                continue
            if task_filter is not None and task_dir.name != task_filter:
                continue
            demos: list[dict] = []
            for traj_dir in sorted(task_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                rec = _build_demo_record(traj_dir)
                if rec is not None:
                    demos.append(rec)
            if demos:
                tasks.append((task_dir.name, demos))
        return tasks

    # Flat fallback.
    demos: list[dict] = []
    for traj_dir in sorted(data_path.iterdir()):
        if not traj_dir.is_dir():
            continue
        rec = _build_demo_record(traj_dir)
        if rec is not None:
            demos.append(rec)
    if not demos:
        return []
    fallback_task = task_filter or data_path.name or "default"
    return [(fallback_task, demos)]


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
        help="Root directory of LfO sample data (must contain training_trajectories/).",
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory; one HDF5 per task is written as <output-dir>/<task>.hdf5.",
    )
    p.add_argument(
        "--task", type=str, default=None,
        help="Optional: only convert this single task name.",
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


def write_task_hdf5(
    task_name: str,
    demos: list[dict],
    output_dir: str,
    camera_key: str,
    img_size: int,
) -> tuple[str, int]:
    """Write one ``<output_dir>/<task_name>.hdf5`` with all demos in this task."""
    out_path = os.path.join(output_dir, f"{task_name}.hdf5")
    os.makedirs(output_dir, exist_ok=True)

    total_samples = 0
    with h5py.File(out_path, "w") as out:
        data_grp = out.create_group("data")
        for i, demo_info in enumerate(demos):
            demo_key = f"demo_{i}"
            print(f"  [{i}] {demo_info['name']} -> {demo_key}")

            obs_data = extract_obs_from_demo(
                demo_info["main_h5"],
                demo_info["state_h5"],
                camera_key,
                img_size,
            )

            T = obs_data["num_samples"]
            total_samples += T

            demo_grp = data_grp.create_group(demo_key)
            demo_grp.attrs["num_samples"] = T

            obs_grp = demo_grp.create_group("obs")
            for key in ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori",
                        "joint_states", "gripper_states"):
                obs_grp.create_dataset(key, data=obs_data[key])

            demo_grp.create_dataset("actions", data=obs_data["actions"])

            dones = np.zeros(T, dtype=np.uint8)
            dones[-1] = 1
            demo_grp.create_dataset("dones", data=dones)
            demo_grp.create_dataset("rewards", data=dones.copy())

        env_args_str = build_env_args(demos[0]["json"] if demos else None)
        data_grp.attrs["env_args"] = env_args_str
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["num_demos"] = len(demos)

    return out_path, total_samples


def main():
    args = parse_args()

    tasks = discover_tasks(args.data_dir, args.task)
    if not tasks:
        print(f"No tasks found in {args.data_dir}")
        return

    print(f"Found {len(tasks)} task(s)")
    for task_name, demos in tasks:
        print(f"\nTask {task_name!r}: {len(demos)} demo(s)")
        out_path, total_samples = write_task_hdf5(
            task_name=task_name,
            demos=demos,
            output_dir=args.output_dir,
            camera_key=args.camera_key,
            img_size=args.img_size,
        )
        print(f"  Wrote {out_path}")
        print(f"    Demos:  {len(demos)}")
        print(f"    Frames: {total_samples}")


if __name__ == "__main__":
    main()
