#!/usr/bin/env python3
"""Convert LfO-benchmark robot trajectories into robomimic-format HDF5s.

Replaces the older ``convert_demo_into_hdf5.py`` which assumed an obsolete
data layout (RGB nested inside the .h5 with a different schema).

Input layout — produced by ``scripts/process_training_trajectories.py``:

    <data-dir>/h5_training_trajectories/<task>/<demo>/<ts>.h5
        traj_0/obs/sensor_data/{base,hand,...}/rgb   (T,   512, 512, 3) uint8
        traj_0/obs/agent/qpos                         (T,   9) float32  # 7 arm + 2 gripper
        traj_0/actions                                (T-1, 8) float32  # pd_joint_pos
        traj_0/{rewards,terminated,truncated,success}
        traj_0/env_states/...

Output — one HDF5 per task, demo index by sorted demo-dir name:

    <output-dir>/<task>.hdf5
        data/
            demo_<i>/
                obs/
                    agentview_rgb     (T-1, 128, 128, 3) uint8  # from base_camera
                    eye_in_hand_rgb   (T-1, 128, 128, 3) uint8  # from hand_camera
                    leftview_rgb      (T-1, 128, 128, 3) uint8  # from left_camera
                    rightview_rgb     (T-1, 128, 128, 3) uint8  # from right_camera
                    joint_states      (T-1, 7) float32
                    gripper_states    (T-1, 2) float32
                actions               (T-1, 8) float32   # pd_joint_pos
                dones                 (T-1,)  uint8      # terminated | truncated
                rewards               (T-1,)  float32
                attrs:
                    num_samples = T-1
            data.attrs:
                total          = sum of demo num_samples
                env_args       = JSON metadata for rollouts (best-effort; from <ts>.json)
        mask/
            train, valid                                 # 90/10 filter keys

Pass ``--leftview-camera ""`` and/or ``--rightview-camera ""`` to drop those
two extra views and fall back to the original 2-view (agentview + eye_in_hand)
output.

Lengths are aligned to action length (T-1) so obs/actions/dones/rewards all
match — robomimic's SequenceDataset requires this.

``ee_pos`` / ``ee_ori`` are intentionally not materialized — our LfO data
doesn't carry ``tcp_pose`` and the per-task ``obs/state`` layout varies. The
companion policy config (``configs/uniskill_policy_lfo.json``) uses joint and
gripper states plus the two RGB streams as observations.

Demo index assignment matches ``extract_skills.py``'s sorted-demo-dir order so
``<skill_dir>/<task>/demo_<i>/`` files line up.

Usage:
    python maniskill/convert_lfo_to_robomimic.py \\
        --data-dir /root/data/lfo_benchmark \\
        --output-dir /root/data/lfo_benchmark/robomimic_hdf5 \\
        [--task pick_cube] [--agentview-camera base_camera] \\
        [--eye-in-hand-camera hand_camera] [--image-size 128]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_AGENTVIEW_CAMERA = "base_camera"
DEFAULT_EYE_IN_HAND_CAMERA = "hand_camera"
DEFAULT_LEFTVIEW_CAMERA = "left_camera"
DEFAULT_RIGHTVIEW_CAMERA = "right_camera"


def _resize_uint8(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize (T, H, W, 3) uint8 to (T, size, size, 3) uint8 with INTER_AREA."""
    T = rgb.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for t in range(T):
        out[t] = cv2.resize(rgb[t], (size, size), interpolation=cv2.INTER_AREA)
    return out


def _iter_demo_h5s(task_dir: Path) -> list[tuple[str, Path]]:
    """Return [(demo_id, h5_path), ...] in sorted-demo-dir order."""
    out: list[tuple[str, Path]] = []
    for demo in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        h5s = [p for p in sorted(demo.glob("*.h5")) if ".state." not in p.name]
        if not h5s:
            continue
        out.append((demo.name, h5s[0]))
    return out


def _read_env_args(h5_path: Path) -> dict | None:
    """Best-effort read of the sidecar JSON next to the h5, returning a dict
    suitable for ``data.attrs["env_args"]`` so rollouts can spin up the env.
    Returns ``None`` if no usable metadata is present.
    """
    sidecar = h5_path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        meta = json.loads(sidecar.read_text())
    except Exception:
        return None
    env_info = meta.get("env_info") or {}
    env_id = env_info.get("env_id")
    if not env_id:
        return None
    return {
        "env_name": env_id,
        "type": "maniskill",
        "env_kwargs": env_info.get("env_kwargs", {}),
    }


def _convert_one_demo(
    h5_path: Path,
    *,
    camera_map: dict,
    image_size: int,
) -> dict:
    """Read one source h5, return a dict of arrays ready to write under ``demo_<i>``.

    All arrays are truncated to ``T - 1`` along the time axis (= action length).

    Args:
        camera_map: Mapping from output obs key (e.g. ``"agentview_rgb"``) to
            source camera name in ``traj_0/obs/sensor_data/<name>/rgb``.
        image_size: target H/W for the cv2.resize downsample.
    """
    with h5py.File(h5_path, "r") as f:
        rgb_arrays = {
            out_key: f[f"traj_0/obs/sensor_data/{src_cam}/rgb"][()]
            for out_key, src_cam in camera_map.items()
        }
        qpos = f["traj_0/obs/agent/qpos"][()]
        actions = f["traj_0/actions"][()]
        rewards = f["traj_0/rewards"][()] if "traj_0/rewards" in f else None
        terminated = f["traj_0/terminated"][()] if "traj_0/terminated" in f else None
        truncated = f["traj_0/truncated"][()] if "traj_0/truncated" in f else None

    T_act = int(actions.shape[0])

    # Align obs to action length: drop the trailing observation (no action follows).
    rgb_small = {
        out_key: _resize_uint8(arr[:T_act], image_size)
        for out_key, arr in rgb_arrays.items()
    }
    qpos = qpos[:T_act]
    joint_states = qpos[:, :7].astype(np.float32)
    gripper_states = qpos[:, 7:9].astype(np.float32)
    actions = actions.astype(np.float32)

    if rewards is None:
        rewards_arr = np.zeros((T_act,), dtype=np.float32)
    else:
        rewards_arr = rewards[:T_act].astype(np.float32)
    if terminated is None and truncated is None:
        dones = np.zeros((T_act,), dtype=np.uint8)
    else:
        term = terminated[:T_act] if terminated is not None else np.zeros((T_act,), dtype=bool)
        trunc = truncated[:T_act] if truncated is not None else np.zeros((T_act,), dtype=bool)
        dones = (term | trunc).astype(np.uint8)

    out = {
        "obs/joint_states": joint_states,
        "obs/gripper_states": gripper_states,
        "actions": actions,
        "rewards": rewards_arr,
        "dones": dones,
        "num_samples": T_act,
    }
    for out_key, arr in rgb_small.items():
        out[f"obs/{out_key}"] = arr
    return out


def _write_task_hdf5(
    task: str,
    demo_h5s: list[tuple[str, Path]],
    out_path: Path,
    *,
    camera_map: dict,
    image_size: int,
    train_frac: float,
    seed: int,
    compression: str | None,
) -> tuple[int, int]:
    """Convert one task's demos into a single robomimic-format HDF5.
    Returns ``(num_demos, num_samples_total)``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    env_args = None
    total_samples = 0
    demo_ids: list[str] = []
    rgb_keys = list(camera_map.keys())  # ordered output keys, e.g. agentview_rgb, eye_in_hand_rgb, leftview_rgb, rightview_rgb
    with h5py.File(tmp_path, "w") as out:
        data_grp = out.create_group("data")
        compression_kw = {"compression": compression} if compression else {}
        for i, (demo_id, h5_path) in enumerate(demo_h5s):
            try:
                arrays = _convert_one_demo(
                    h5_path,
                    camera_map=camera_map,
                    image_size=image_size,
                )
            except (KeyError, OSError) as e:
                print(f"  skip {task}/{demo_id}: {e}")
                continue
            demo_grp = data_grp.create_group(f"demo_{i}")
            demo_grp.attrs["num_samples"] = int(arrays["num_samples"])
            obs_grp = demo_grp.create_group("obs")
            for rgb_key in rgb_keys:
                obs_grp.create_dataset(
                    rgb_key,
                    data=arrays[f"obs/{rgb_key}"],
                    chunks=(1, image_size, image_size, 3),
                    **compression_kw,
                )
            obs_grp.create_dataset("joint_states", data=arrays["obs/joint_states"])
            obs_grp.create_dataset("gripper_states", data=arrays["obs/gripper_states"])
            demo_grp.create_dataset("actions", data=arrays["actions"])
            demo_grp.create_dataset("rewards", data=arrays["rewards"])
            demo_grp.create_dataset("dones", data=arrays["dones"])
            total_samples += int(arrays["num_samples"])
            demo_ids.append(demo_grp.name.split("/")[-1])
            if env_args is None:
                env_args = _read_env_args(h5_path)

        data_grp.attrs["total"] = total_samples
        if env_args is not None:
            data_grp.attrs["env_args"] = json.dumps(env_args)

        # 90/10 deterministic mask split (rng seeded by task name + seed; mask
        # to 64-bit non-negative since numpy's default_rng rejects negatives).
        if demo_ids:
            rng = np.random.default_rng((hash(task) ^ seed) & 0xFFFFFFFFFFFFFFFF)
            idxs = np.arange(len(demo_ids))
            rng.shuffle(idxs)
            cut = max(1, int(round(len(idxs) * train_frac)))
            train_ids = sorted(demo_ids[j] for j in idxs[:cut])
            valid_ids = sorted(demo_ids[j] for j in idxs[cut:])
            mask_grp = out.create_group("mask")
            mask_grp.create_dataset(
                "train", data=np.array([n.encode("utf-8") for n in train_ids])
            )
            mask_grp.create_dataset(
                "valid", data=np.array([n.encode("utf-8") for n in valid_ids])
            )

    os.replace(tmp_path, out_path)
    return len(demo_ids), total_samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data-dir",
        required=True,
        help="LfO benchmark root containing h5_training_trajectories/.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the per-task robomimic HDF5s.",
    )
    p.add_argument(
        "--input-subdir",
        default="h5_training_trajectories",
        help="Subdir under data-dir holding the processed sensor_data-mode h5s.",
    )
    p.add_argument(
        "--task",
        default=None,
        help="If set, only convert this task. Otherwise iterate all tasks under input-subdir.",
    )
    p.add_argument(
        "--agentview-camera",
        default=DEFAULT_AGENTVIEW_CAMERA,
        help=f"Camera key for obs/agentview_rgb (default: {DEFAULT_AGENTVIEW_CAMERA}).",
    )
    p.add_argument(
        "--eye-in-hand-camera",
        default=DEFAULT_EYE_IN_HAND_CAMERA,
        help=f"Camera key for obs/eye_in_hand_rgb (default: {DEFAULT_EYE_IN_HAND_CAMERA}).",
    )
    p.add_argument(
        "--leftview-camera",
        default=DEFAULT_LEFTVIEW_CAMERA,
        help=f"Camera key for obs/leftview_rgb. Pass empty string to omit "
             f"(default: {DEFAULT_LEFTVIEW_CAMERA}).",
    )
    p.add_argument(
        "--rightview-camera",
        default=DEFAULT_RIGHTVIEW_CAMERA,
        help=f"Camera key for obs/rightview_rgb. Pass empty string to omit "
             f"(default: {DEFAULT_RIGHTVIEW_CAMERA}).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Downsampled image side length for the robomimic obs (default: 128).",
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.9,
        help="Fraction of demos in the train mask filter (default: 0.9).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the train/valid mask split (default: 0).",
    )
    p.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="gzip",
        help="HDF5 compression for image datasets (default: gzip).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-task HDF5s.",
    )
    args = p.parse_args()

    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    out_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    src_root = data_dir / args.input_subdir
    if not src_root.is_dir():
        raise FileNotFoundError(f"{src_root} not found")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task is not None:
        tasks = [args.task]
    else:
        tasks = sorted(p.name for p in src_root.iterdir() if p.is_dir())
    if not tasks:
        raise SystemExit(f"No task dirs under {src_root}")

    compression = None if args.compression == "none" else args.compression

    # Build the output_key -> source_camera mapping. Empty strings disable a view.
    camera_map = {}
    for out_key, src_cam in [
        ("agentview_rgb", args.agentview_camera),
        ("eye_in_hand_rgb", args.eye_in_hand_camera),
        ("leftview_rgb", args.leftview_camera),
        ("rightview_rgb", args.rightview_camera),
    ]:
        if src_cam:
            camera_map[out_key] = src_cam
    if not camera_map:
        raise SystemExit("At least one camera must be enabled.")

    print(f"converting {len(tasks)} task(s) → {out_dir}  (rgb keys: {list(camera_map)})")
    for task in tasks:
        task_dir = src_root / task
        if not task_dir.is_dir():
            print(f"  skip {task}: missing source dir")
            continue
        out_path = out_dir / f"{task}.hdf5"
        if out_path.exists() and not args.overwrite:
            print(f"  skip {task}: {out_path} already exists (pass --overwrite to redo)")
            continue
        demos = _iter_demo_h5s(task_dir)
        if not demos:
            print(f"  skip {task}: no source demos under {task_dir}")
            continue
        n, total = _write_task_hdf5(
            task,
            demos,
            out_path,
            camera_map=camera_map,
            image_size=args.image_size,
            train_frac=args.train_frac,
            seed=args.seed,
            compression=compression,
        )
        print(f"  {task}: wrote {n} demos / {total} samples → {out_path}")


if __name__ == "__main__":
    main()
