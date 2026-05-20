#!/usr/bin/env python3
"""Convert LfO-benchmark robot trajectories into robomimic-format HDF5s.

Replaces the older ``convert_demo_into_hdf5.py`` which assumed an obsolete
data layout (RGB nested inside the .h5 with a different schema).

Input layout — produced by ``scripts/process_training_trajectories.py``:

    <data-dir>/postprocessed_robot_trajectories/
        data/<idx>.pkl          one per (task, demo, timestep) sample, holding
                                observation/<camera> (H, W, 3) uint8,
                                qpos (9,) float32  # 7 arm + 2 gripper,
                                actions (action_horizon, 8) float32,
                                task / demo_id / robot_time_index
        meta/index.jsonl        one JSON line per pkl: {task, demo_id, t, idx}

(The legacy ``h5_training_trajectories/<task>/<demo>/<ts>.h5`` input was
removed when ``process_training_trajectories.py`` switched to the per-sample
pkl format; this script was migrated to read the pkls directly.)

Output — one HDF5 per task, demo index by sorted ``demo_id``. Default emits
the **2-view** (front + wrist) configuration. Pass ``--leftview-camera`` /
``--rightview-camera`` to add side views (only if the pkls carry them).

    <output-dir>/<task>.hdf5
        data/
            demo_<i>/
                obs/
                    agentview_rgb     (T, 128, 128, 3) uint8  # from base_camera (front)
                    eye_in_hand_rgb   (T, 128, 128, 3) uint8  # from hand_camera (wrist)
                    joint_states      (T, 7) float32
                    gripper_states    (T, 2) float32
                actions               (T, 8) float32   # pd_joint_pos
                dones                 (T,)  uint8
                rewards               (T,)  float32
                attrs:
                    num_samples = T
            data.attrs:
                total          = sum of demo num_samples
        mask/
            train, valid                                 # 90/10 filter keys

``T`` is the number of per-sample pkls for a demo (one per valid action-window
start timestep). obs/actions/dones/rewards all share that length —
robomimic's SequenceDataset requires this.

``ee_pos`` / ``ee_ori`` are intentionally not materialized — our LfO data
doesn't carry ``tcp_pose``. The companion policy config
(``configs/uniskill_policy_lfo.json``) uses joint and gripper states plus the
two RGB streams as observations.

Demo index assignment matches ``scripts/extract_uniskill_robot_skill.py``'s
sorted-``demo_id`` order (both group ``meta/index.jsonl`` and sort the keys)
so ``<skill_dir>/<task>/demo_<i>/`` files line up.

Usage:
    python maniskill/convert_lfo_to_robomimic.py \\
        --data-dir /root/data/lfo_benchmark \\
        --output-dir /root/data/lfo_benchmark/robomimic_hdf5 \\
        [--task pick_cube] [--agentview-camera base_camera] \\
        [--eye-in-hand-camera hand_camera] [--image-size 128]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_AGENTVIEW_CAMERA = "base_camera"
DEFAULT_EYE_IN_HAND_CAMERA = "hand_camera"
# Empty defaults — left/right views are off by default. Pass --leftview-camera left_camera
# (and similarly --rightview-camera right_camera) to enable them.
DEFAULT_LEFTVIEW_CAMERA = ""
DEFAULT_RIGHTVIEW_CAMERA = ""


def _resize_uint8(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize (T, H, W, 3) uint8 to (T, size, size, 3) uint8 with INTER_AREA."""
    T = rgb.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for t in range(T):
        out[t] = cv2.resize(rgb[t], (size, size), interpolation=cv2.INTER_AREA)
    return out


#: ManiSkill articulation row layout: [root_pose(7), root_lin_vel(3),
#: root_ang_vel(3), qpos(N), qvel(N)]. For the 9-DoF panda this is
#: 13 + 9 + 9 = 31 columns; qpos lives at offset 13 with length 9.
_ARTICULATION_QPOS_OFFSET = 13
_ARTICULATION_QPOS_LEN = 9


def _discover_demos(
    samples_root: Path, raw_root: Path, task_filter: str | None
) -> list[tuple[str, list[tuple[str, list[int], str, Path]]]]:
    """Group sample idxs by ``(task, demo_id)`` from ``meta/index.jsonl`` and
    attach the demo's raw-trajectory location.

    Returns ``[(task, [(demo_id, [sample_idx sorted by t, ...], h5_path,
    demo_dir), ...]), ...]`` with ``demo_id`` keys sorted. ``h5_path`` /
    ``demo_dir`` point into ``raw_training_trajectories/<task>/<demo_id>/`` so
    :func:`_convert_one_demo` can fill in the trailing ``action_horizon``
    timesteps that the pi05 trim drops from the pkls.
    """
    import glob
    index_path = samples_root / "meta" / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Expected {index_path}. Run scripts/process_training_trajectories.py first."
        )
    groups: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            groups[e["task"]][e["demo_id"]].append((int(e["t"]), int(e["idx"])))

    tasks: list[tuple[str, list[tuple[str, list[int], str, Path]]]] = []
    for task in sorted(groups):
        if task_filter is not None and task != task_filter:
            continue
        demos: list[tuple[str, list[int], str, Path]] = []
        for demo_id in sorted(groups[task]):
            sample_idxs = [idx for _t, idx in sorted(groups[task][demo_id])]
            demo_dir = raw_root / task / demo_id
            h5s = sorted(glob.glob(str(demo_dir / "*.h5")))
            if not h5s:
                print(f"  warn: no h5 for {task}/{demo_id} under {demo_dir}; "
                      "writing pkl-trimmed length only")
                h5_path = ""
            else:
                h5_path = h5s[0]
            demos.append((demo_id, sample_idxs, h5_path, demo_dir))
        if demos:
            tasks.append((task, demos))
    return tasks


def _convert_one_demo(
    samples_root: Path,
    sample_idxs: list[int],
    *,
    camera_map: dict,
    image_size: int,
    h5_path: str = "",
    demo_dir: Path | None = None,
) -> dict:
    """Read one demo's per-sample pkls, return arrays ready to write under ``demo_<i>``.

    Args:
        camera_map: Mapping from output obs key (e.g. ``"agentview_rgb"``) to
            the source camera name stored as ``observation/<name>`` in each pkl.
        image_size: target H/W for the cv2.resize downsample.
        h5_path: optional raw-trajectory ``*.h5`` path. When given alongside
            ``demo_dir``, the converter fills in the trailing
            ``T_act - len(sample_idxs)`` timesteps that the pi05 horizon trim
            drops — actions + qpos from the h5, images from the demo's mp4s —
            so the output HDF5 covers the *full* trajectory (length ``T_act``).
        demo_dir: ``raw_training_trajectories/<task>/<demo>/`` (needed to
            locate ``<src_cam>.mp4`` for the trailing-frame fill-in).
    """
    data_dir = samples_root / "data"
    rgb_acc: dict[str, list[np.ndarray]] = {k: [] for k in camera_map}
    joint_acc: list[np.ndarray] = []
    gripper_acc: list[np.ndarray] = []
    action_acc: list[np.ndarray] = []

    for idx in sample_idxs:
        with open(data_dir / f"{idx}.pkl", "rb") as fh:
            s = pickle.load(fh)
        for out_key, src_cam in camera_map.items():
            rgb_acc[out_key].append(np.asarray(s[f"observation/{src_cam}"]))
        qpos = np.asarray(s["qpos"], dtype=np.float32)
        joint_acc.append(qpos[:7])
        gripper_acc.append(qpos[7:9])
        # The pkl stores an action window actions[t : t+horizon]; the action
        # *at* timestep t — what robomimic wants per step — is window row 0.
        action_acc.append(np.asarray(s["actions"], dtype=np.float32)[0])

    # Fill in the trailing timesteps that the pi05 horizon trim drops from
    # the pkls (UniSkill-Policy is single-step; pi05 chunk size is irrelevant).
    if h5_path and demo_dir is not None:
        with h5py.File(h5_path, "r") as f:
            T_act = int(f["traj_0/actions"].shape[0])
            extra = T_act - len(sample_idxs)
            if extra > 0:
                tail_actions = np.asarray(
                    f["traj_0/actions"][len(sample_idxs):T_act], dtype=np.float32)
                art_keys = list(f["traj_0/env_states/articulations"].keys())
                robot_key = next((k for k in art_keys if "panda" in k.lower()),
                                 art_keys[0])
                art = np.asarray(
                    f[f"traj_0/env_states/articulations/{robot_key}"][
                        len(sample_idxs):T_act], dtype=np.float32)
                tail_qpos = art[:, _ARTICULATION_QPOS_OFFSET :
                                _ARTICULATION_QPOS_OFFSET + _ARTICULATION_QPOS_LEN]
                for row in range(extra):
                    action_acc.append(tail_actions[row])
                    joint_acc.append(tail_qpos[row, :7])
                    gripper_acc.append(tail_qpos[row, 7:9])
            else:
                extra = 0
        # Pull the trailing camera frames from the per-camera mp4s (frame index
        # == timestep, verified: cam_frames == T_act). Raw mp4 frames are
        # 512x512 while the existing pkl frames are 224x224 (preprocessed in
        # process_training_trajectories.py); resize the new frames to match
        # before stacking so np.stack accepts the concatenation.
        if extra > 0:
            from decord import VideoReader, cpu
            pkl_shape = rgb_acc[next(iter(camera_map))][0].shape  # (Hpkl, Wpkl, 3)
            tgt_h, tgt_w = pkl_shape[0], pkl_shape[1]
            tail_idxs = list(range(len(sample_idxs), len(sample_idxs) + extra))
            for out_key, src_cam in camera_map.items():
                mp4_path = demo_dir / f"{src_cam}.mp4"
                if not mp4_path.is_file():
                    raise FileNotFoundError(
                        f"trailing-frame fill needs {mp4_path} (found h5 only)")
                vr = VideoReader(str(mp4_path), ctx=cpu(0))
                idxs_in_vr = [min(t, len(vr) - 1) for t in tail_idxs]
                frames = vr.get_batch(idxs_in_vr).asnumpy()  # (extra, H, W, 3)
                for fr in frames:
                    if (fr.shape[0], fr.shape[1]) != (tgt_h, tgt_w):
                        fr = cv2.resize(fr, (tgt_w, tgt_h),
                                        interpolation=cv2.INTER_AREA)
                    rgb_acc[out_key].append(fr)

    T = len(action_acc)
    out: dict = {
        "obs/joint_states": np.stack(joint_acc, axis=0).astype(np.float32),
        "obs/gripper_states": np.stack(gripper_acc, axis=0).astype(np.float32),
        "actions": np.stack(action_acc, axis=0).astype(np.float32),
        # The per-sample pkls don't carry rewards/terminated/truncated; the
        # UniSkill policy trains offline (no rollouts), so zeros are fine and
        # match the legacy converter's missing-metadata path.
        "rewards": np.zeros((T,), dtype=np.float32),
        "dones": np.zeros((T,), dtype=np.uint8),
        "num_samples": T,
    }
    for out_key, frames in rgb_acc.items():
        out[f"obs/{out_key}"] = _resize_uint8(np.stack(frames, axis=0), image_size)
    return out


def _write_task_hdf5(
    task: str,
    samples_root: Path,
    demos: list[tuple[str, list[int], str, Path]],
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

    total_samples = 0
    demo_ids: list[str] = []
    rgb_keys = list(camera_map.keys())  # ordered output keys, e.g. agentview_rgb, eye_in_hand_rgb
    with h5py.File(tmp_path, "w") as out:
        data_grp = out.create_group("data")
        compression_kw = {"compression": compression} if compression else {}
        for i, (demo_id, sample_idxs, h5_path, demo_dir) in enumerate(demos):
            try:
                arrays = _convert_one_demo(
                    samples_root,
                    sample_idxs,
                    camera_map=camera_map,
                    image_size=image_size,
                    h5_path=h5_path,
                    demo_dir=demo_dir,
                )
            except (KeyError, OSError, ValueError) as e:
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

        data_grp.attrs["total"] = total_samples
        # robomimic's get_env_metadata_from_dataset unconditionally reads
        # data.attrs["env_args"]; rollouts are disabled (out of scope) so the
        # contents are unused, but the attribute must exist or train() raises
        # KeyError before training starts.
        data_grp.attrs["env_args"] = json.dumps(
            {"env_name": task, "type": "maniskill", "env_kwargs": {}}
        )

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
        help="LfO benchmark root containing postprocessed_robot_trajectories/.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the per-task robomimic HDF5s.",
    )
    p.add_argument(
        "--input-subdir",
        default="postprocessed_robot_trajectories",
        help="Subdir under data-dir holding the per-sample robot-trajectory pkls "
             "+ meta/index.jsonl (default: postprocessed_robot_trajectories).",
    )
    p.add_argument(
        "--raw-subdir",
        default="raw_training_trajectories",
        help="Subdir under data-dir holding the raw per-demo trajectory "
             "(<task>/<demo>/*.h5 + per-camera mp4s). Used to fill in the "
             "trailing T_act - pkl_count timesteps the pi05 horizon trim drops "
             "(default: raw_training_trajectories).",
    )
    p.add_argument(
        "--task",
        default=None,
        help="If set, only convert this task. Otherwise iterate all tasks in index.jsonl.",
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
    raw_root = data_dir / args.raw_subdir
    if not src_root.is_dir():
        raise FileNotFoundError(f"{src_root} not found")
    if not raw_root.is_dir():
        print(f"  warn: {raw_root} not found — trailing-frame fill-in disabled")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = _discover_demos(src_root, raw_root, args.task)
    if not tasks:
        raise SystemExit(f"No demos found under {src_root}")

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
    for task, demos in tasks:
        out_path = out_dir / f"{task}.hdf5"
        if out_path.exists() and not args.overwrite:
            print(f"  skip {task}: {out_path} already exists (pass --overwrite to redo)")
            continue
        n, total = _write_task_hdf5(
            task,
            src_root,
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
