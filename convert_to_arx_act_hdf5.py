#!/usr/bin/env python3
"""
将 Flexiv 对齐后的 episode.hdf5 转为 ARX mobile_aloha ACT 训练可读格式。

ARX 侧读取逻辑见: ARX_PLAY_plus/mobile_aloha/utils/utils.py 中 EpisodicDataset。

用法示例:
  # 单个 episode 目录
  python convert_to_arx_act_hdf5.py \\
    --flexiv_episode dataset/pick_cup/episode_000 \\
    --out_dir ./arx_act_dataset \\
    --episode_index 0

  # 整个 task 下所有 episode_*
  python convert_to_arx_act_hdf5.py \\
    --flexiv_task dataset/pick_cup \\
    --out_dir ./arx_act_dataset

  # 四路鱼眼（与 train.py --camera_names left_cam0 left_cam1 right_cam0 right_cam1 对齐）
  python convert_to_arx_act_hdf5.py \\
    --flexiv_task dataset/pick_cup --out_dir ./arx_act_dataset --four_fisheyes
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


# 四路鱼眼时 HDF5 里 observations/images 的 key 与 train.py --camera_names 一致即可（常用：与 Flexiv 同名）
FLEXIV_FOUR_CAM_KEYS = ["left_cam0", "left_cam1", "right_cam0", "right_cam1"]


def _write_one(
    flexiv_h5: Path,
    out_path: Path,
    cam_sources: dict[str, str],
) -> None:
    """flexiv_h5: episode.hdf5 from build_dataset.py.

    cam_sources: ARX 图像 dataset 名 -> Flexiv camera/* 子键名（通常四路时二者相同）。
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))

    with h5py.File(flexiv_h5, "r") as src, h5py.File(out_path, "w") as dst:
        left_q = np.asarray(src["left_arm/q"], dtype=np.float32)
        right_q = np.asarray(src["right_arm/q"], dtype=np.float32)
        left_tcp = np.asarray(src["left_arm/tcp_pose"], dtype=np.float32)
        right_tcp = np.asarray(src["right_arm/tcp_pose"], dtype=np.float32)
        n = left_q.shape[0]
        if n == 0:
            raise ValueError(f"empty episode: {flexiv_h5}")

        qpos = np.concatenate([left_q, right_q], axis=1)
        eef = np.concatenate([left_tcp, right_tcp], axis=1)
        qvel = np.zeros_like(qpos, dtype=np.float32)
        effort = np.zeros_like(qpos, dtype=np.float32)
        robot_base = np.zeros((n, 9), dtype=np.float32)
        action = qpos.copy()

        dst.attrs["sim"] = np.bool_(False)
        dst.attrs["compress"] = np.bool_(True)

        dst.create_dataset("action", data=action.astype(np.float32), compression="gzip", compression_opts=1)

        obs = dst.create_group("observations")
        obs.create_dataset("qpos", data=qpos.astype(np.float32), compression="gzip", compression_opts=1)
        obs.create_dataset("eef", data=eef.astype(np.float32), compression="gzip", compression_opts=1)
        obs.create_dataset("qvel", data=qvel, compression="gzip", compression_opts=1)
        obs.create_dataset("effort", data=effort, compression="gzip", compression_opts=1)
        obs.create_dataset("robot_base", data=robot_base, compression="gzip", compression_opts=1)

        img_grp = obs.create_group("images")
        for arx_name, flex_name in cam_sources.items():
            ds_src = src[f"camera/{flex_name}"]
            d = img_grp.create_dataset(arx_name, (n,), dtype=vlen_u8)
            for i in range(n):
                raw = ds_src[i]
                arr = np.asarray(raw, dtype=np.uint8).ravel()
                d[i] = arr


def main():
    p = argparse.ArgumentParser(description="Flexiv episode.hdf5 -> ARX ACT episode_N.hdf5")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--flexiv_episode", type=str, help="单集目录，内含 episode.hdf5")
    g.add_argument("--flexiv_task", type=str, help="task 目录，其下 episode_*/episode.hdf5 批量转换")

    p.add_argument("--out_dir", type=str, required=True, help="输出目录（平铺 episode_0.hdf5, episode_1.hdf5, ...）")
    p.add_argument(
        "--episode_index",
        type=int,
        default=None,
        help="仅单集模式: 输出文件名 episode_{index}.hdf5（默认 0）",
    )
    p.add_argument(
        "--head_cam",
        type=str,
        default="left_cam0",
        help="Flexiv camera/* 中映射到 ARX 的 head",
    )
    p.add_argument(
        "--left_wrist_cam",
        type=str,
        default="left_cam1",
        help="映射到 left_wrist",
    )
    p.add_argument(
        "--right_wrist_cam",
        type=str,
        default="right_cam0",
        help="映射到 right_wrist",
    )
    p.add_argument(
        "--four_fisheyes",
        action="store_true",
        help="写入四路相机 observations/images/{left_cam0,left_cam1,right_cam0,right_cam1}（与 Flexiv 键名一致）；"
        "训练时: --camera_names left_cam0 left_cam1 right_cam0 right_cam1",
    )
    p.add_argument("--overwrite_out_dir", action="store_true", help="若存在 out_dir 则先删除再写入（慎用）")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite_out_dir:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.four_fisheyes:
        cam_sources = {k: k for k in FLEXIV_FOUR_CAM_KEYS}
    else:
        cam_sources = {
            "head": args.head_cam,
            "left_wrist": args.left_wrist_cam,
            "right_wrist": args.right_wrist_cam,
        }

    if args.flexiv_episode:
        ep_dir = Path(args.flexiv_episode)
        h5 = ep_dir / "episode.hdf5"
        if not h5.is_file():
            raise FileNotFoundError(h5)
        idx = 0 if args.episode_index is None else args.episode_index
        out = out_dir / f"episode_{idx}.hdf5"
        _write_one(h5, out, cam_sources)
        print(f"Wrote {out}")
        return

    task = Path(args.flexiv_task)
    ep_dirs = sorted([d for d in task.glob("episode_*") if d.is_dir() and (d / "episode.hdf5").is_file()])
    if not ep_dirs:
        raise FileNotFoundError(f"no episode_*/episode.hdf5 under {task}")

    for i, d in enumerate(ep_dirs):
        _write_one(
            d / "episode.hdf5",
            out_dir / f"episode_{i}.hdf5",
            cam_sources,
        )
        print(f"episode_{i}.hdf5  <-  {d.name}")

    print(f"Done: {len(ep_dirs)} files in {out_dir}")


if __name__ == "__main__":
    main()
