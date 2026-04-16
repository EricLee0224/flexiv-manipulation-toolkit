#!/usr/bin/env python3
"""
检查与 train.py --gripper_binary 一致的夹爪处理：全局 1%/99% 分位 remap → [0,1]，再 0.5 二值化。

对数据集中每条 episode 打印统计，并可选为每条 episode 保存曲线图（raw / [0,1] / binary）。

用法:
  # Flexiv build 输出: task/episode_000/episode.hdf5 …
  python diagnose_gripper_binary_training.py \\
    --dataset_dir /path/to/flexiv-manipulation-toolkit/dataset/place_tube_150 \\
    --plots_dir ./gripper_binary_check_plots

  # ACT 平铺: task/episode_0.hdf5 …
  python diagnose_gripper_binary_training.py \\
    --dataset_dir /path/to/mobile_aloha/dataset/place_tube_150 \\
    --num_episodes 150 --plots_dir /tmp/gripper_plots

  # 仅连续 remap [0,1]（与 train --gripper_remap_01 一致），不画二值行
  python diagnose_gripper_binary_training.py \\
    --dataset_dir /path/to/dataset/0413_place_tube_250 \\
    --plots_dir ./gripper_remap01_state_plots --remap_01_only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

JOINTS_PER_ARM = 8
P_LO, P_HI = 1.0, 99.0


def discover_flexiv_episodes(dataset_dir: Path) -> list[tuple[str, Path]]:
    """episode_*/episode.hdf5 under task dir, sorted by name."""
    out: list[tuple[str, Path]] = []
    for d in sorted(dataset_dir.glob("episode_*")):
        if not d.is_dir():
            continue
        h5 = d / "episode.hdf5"
        if h5.is_file():
            out.append((d.name, h5))
    return out


def scan_percentiles_act(dataset_dir: Path, num_episodes: int, j: int = JOINTS_PER_ARM):
    """ACT: qpos + action 并集（与 utils.scan_gripper_min_max 一致）。"""
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for i in range(num_episodes):
        p = dataset_dir / f"episode_{i}.hdf5"
        if not p.is_file():
            continue
        with h5py.File(p, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
        for arr in (qpos, action):
            left_parts.append(arr[:, j - 1].astype(np.float64))
            right_parts.append(arr[:, 2 * j - 1].astype(np.float64))
    if not left_parts:
        return None
    lv = np.concatenate(left_parts)
    rv = np.concatenate(right_parts)
    return (
        float(np.percentile(lv, P_LO)),
        float(np.percentile(lv, P_HI)),
        float(np.percentile(rv, P_LO)),
        float(np.percentile(rv, P_HI)),
    )


def scan_percentiles_flexiv(episodes: list[tuple[str, Path]]):
    """Flexiv build: left_arm/gripper + right_arm/gripper 全 task 拼接。"""
    parts_l: list[np.ndarray] = []
    parts_r: list[np.ndarray] = []
    for _name, h5 in episodes:
        with h5py.File(h5, "r") as root:
            parts_l.append(np.asarray(root["left_arm/gripper"][:], dtype=np.float64))
            parts_r.append(np.asarray(root["right_arm/gripper"][:], dtype=np.float64))
    if not parts_l:
        return None
    lv = np.concatenate(parts_l)
    rv = np.concatenate(parts_r)
    return (
        float(np.percentile(lv, P_LO)),
        float(np.percentile(lv, P_HI)),
        float(np.percentile(rv, P_LO)),
        float(np.percentile(rv, P_HI)),
    )


def detect_layout(dataset_dir: Path) -> str:
    """'flexiv' | 'act' | unknown"""
    if discover_flexiv_episodes(dataset_dir):
        return "flexiv"
    if (dataset_dir / "episode_0.hdf5").is_file():
        with h5py.File(dataset_dir / "episode_0.hdf5", "r") as f:
            if "/observations/qpos" in f:
                return "act"
    return "unknown"


def remap_01(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    r = max(hi - lo, 1e-8)
    return np.clip((x.astype(np.float64) - lo) / r, 0.0, 1.0)


def binarize01(u: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (u > thr).astype(np.float64)


def count_transitions(b: np.ndarray) -> int:
    if len(b) < 2:
        return 0
    return int(np.sum(b[1:] != b[:-1]))


def plot_episode(
    out_path: Path,
    t: np.ndarray,
    raw_l: np.ndarray,
    raw_r: np.ndarray,
    u_l: np.ndarray,
    u_r: np.ndarray,
    b_l: np.ndarray,
    b_r: np.ndarray,
    lmin: float,
    lmax: float,
    rmin: float,
    rmax: float,
    ep_name: str,
):
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"{ep_name}  |  remap L:[{lmin:.4f},{lmax:.4f}] R:[{rmin:.4f},{rmax:.4f}]  thr=0.5",
        fontsize=11,
        fontweight="bold",
    )
    for col, (side, raw, u, b) in enumerate(
        [("Left", raw_l, u_l, b_l), ("Right", raw_r, u_r, b_r)]
    ):
        axes[0, col].plot(t, raw, "k-", lw=0.8)
        axes[0, col].set_ylabel("raw qpos")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_title(side)

        axes[1, col].plot(t, u, "C0-", lw=0.8)
        axes[1, col].axhline(0.5, color="r", ls="--", lw=0.8)
        axes[1, col].set_ylim(-0.05, 1.05)
        axes[1, col].set_ylabel("remap [0,1]")
        axes[1, col].grid(True, alpha=0.3)

        axes[2, col].step(t, b, where="post", color="C2", lw=0.9)
        axes[2, col].set_ylim(-0.1, 1.1)
        axes[2, col].set_ylabel("binary")
        axes[2, col].set_xlabel("step")
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_episode_remap_01_only(
    out_path: Path,
    t: np.ndarray,
    raw_l: np.ndarray,
    raw_r: np.ndarray,
    u_l: np.ndarray,
    u_r: np.ndarray,
    lmin: float,
    lmax: float,
    rmin: float,
    rmax: float,
    ep_name: str,
):
    """State 夹爪：raw + 线性 remap 到 [0,1]（同 --gripper_remap_01），无二值化曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.5), sharex=True)
    fig.suptitle(
        f"{ep_name}  |  state gripper: raw -> remap [0,1]  "
        f"L:[{lmin:.4f},{lmax:.4f}] R:[{rmin:.4f},{rmax:.4f}]  "
        f"(global {P_LO}%/{P_HI}% pct, same as train --gripper_remap_01)",
        fontsize=10,
        fontweight="bold",
    )
    for col, (side, raw, u) in enumerate(
        [("Left", raw_l, u_l), ("Right", raw_r, u_r)]
    ):
        axes[0, col].plot(t, raw, "k-", lw=0.8)
        axes[0, col].set_ylabel("raw (qpos / gripper)")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_title(side)

        axes[1, col].plot(t, u, "C0-", lw=0.8)
        axes[1, col].axhline(0.5, color="r", ls="--", lw=0.7, alpha=0.8)
        axes[1, col].set_ylim(-0.05, 1.05)
        axes[1, col].set_ylabel("remap [0,1]")
        axes[1, col].set_xlabel("step")
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    workspace = here.parent
    candidates = [
        here / "dataset" / "place_tube_150",
        workspace / "ARX_PLAY_plus" / "mobile_aloha" / "dataset" / "place_tube_150",
    ]
    default_ds = next((c for c in candidates if c.is_dir()), None)

    ap = argparse.ArgumentParser(description="Diagnose gripper remap+binary vs train.py --gripper_binary")
    ap.add_argument(
        "--dataset_dir",
        type=Path,
        default=default_ds,
        help="Flexiv: task 目录（episode_*/episode.hdf5）；ACT: 平铺 episode_N.hdf5。默认 toolkit/dataset/place_tube_150",
    )
    ap.add_argument(
        "--num_episodes",
        type=int,
        default=150,
        help="仅 ACT 模式：扫描 episode_0 … episode_{N-1}。Flexiv 模式自动枚举全部 episode_*",
    )
    ap.add_argument(
        "--plots_dir",
        type=Path,
        default=None,
        help="若指定，为每条 episode 写入 episode_NNN.png（raw / remap / binary）",
    )
    ap.add_argument(
        "--remap_01_only",
        action="store_true",
        help="图与 train --gripper_remap_01 一致：只画 state 夹爪 raw + remap[0,1] 两行（不二值化）。需配合 --plots_dir",
    )
    args = ap.parse_args()

    if args.dataset_dir is None or not args.dataset_dir.is_dir():
        print("请指定存在的 --dataset_dir")
        sys.exit(1)

    ds = args.dataset_dir.resolve()
    layout = detect_layout(ds)
    if layout == "unknown":
        print(f"无法识别数据布局: {ds}\n"
              "  需要 Flexiv: episode_*/episode.hdf5\n"
              "  或 ACT: episode_0.hdf5 且含 observations/qpos")
        sys.exit(1)

    if layout == "flexiv":
        flex_eps = discover_flexiv_episodes(ds)
        mm = scan_percentiles_flexiv(flex_eps)
        scan_note = (
            f"Flexiv HDF5（left_arm/gripper + right_arm/gripper，{len(flex_eps)} 条 episode）"
        )
    else:
        mm = scan_percentiles_act(ds, args.num_episodes)
        scan_note = "ACT（qpos+action 并集，同 utils.scan_gripper_min_max）"

    if mm is None:
        print(f"未找到可用 episode under {ds}")
        sys.exit(1)
    lmin, lmax, rmin, rmax = mm
    thr_raw_l = (lmin + lmax) / 2.0
    thr_raw_r = (rmin + rmax) / 2.0

    print(f"布局: {layout}  |  {scan_note}")
    print(
        f"全局 {P_LO}% / {P_HI}% 分位 remap 区间:\n"
        f"  Left  [{lmin:.6f}, {lmax:.6f}]  → 0.5 对应 raw ≈ {thr_raw_l:.6f}\n"
        f"  Right [{rmin:.6f}, {rmax:.6f}]  → 0.5 对应 raw ≈ {thr_raw_r:.6f}\n"
    )

    if args.plots_dir is not None and plt is None:
        print("未安装 matplotlib，忽略 --plots_dir")
        args.plots_dir = None

    if args.remap_01_only and args.plots_dir is None:
        print("提示: --remap_01_only 需配合 --plots_dir 才会出图；仍打印统计。")

    ok = 0
    if layout == "flexiv":
        for ep_name, p in flex_eps:
            with h5py.File(p, "r") as root:
                raw_l = np.asarray(root["left_arm/gripper"][:], dtype=np.float64)
                raw_r = np.asarray(root["right_arm/gripper"][:], dtype=np.float64)
            u_l = remap_01(raw_l, lmin, lmax)
            u_r = remap_01(raw_r, rmin, rmax)
            b_l = binarize01(u_l)
            b_r = binarize01(u_r)
            n = len(raw_l)
            t = np.arange(n)

            n_open_l = int(np.sum(b_l == 0))
            n_close_l = int(np.sum(b_l == 1))
            n_open_r = int(np.sum(b_r == 0))
            n_close_r = int(np.sum(b_r == 1))

            print(
                f"{ep_name}: n={n}  "
                f"L raw[{raw_l.min():.4f},{raw_l.max():.4f}] "
                f"u[{u_l.min():.3f},{u_l.max():.3f}] "
                f"open={n_open_l} close={n_close_l} trans={count_transitions(b_l)}  |  "
                f"R raw[{raw_r.min():.4f},{raw_r.max():.4f}] "
                f"u[{u_r.min():.3f},{u_r.max():.3f}] "
                f"open={n_open_r} close={n_close_r} trans={count_transitions(b_r)}"
            )

            if args.plots_dir is not None:
                if args.remap_01_only:
                    plot_episode_remap_01_only(
                        args.plots_dir / f"{ep_name}.png",
                        t, raw_l, raw_r, u_l, u_r,
                        lmin, lmax, rmin, rmax, ep_name,
                    )
                else:
                    plot_episode(
                        args.plots_dir / f"{ep_name}.png",
                        t,
                        raw_l,
                        raw_r,
                        u_l,
                        u_r,
                        b_l,
                        b_r,
                        lmin,
                        lmax,
                        rmin,
                        rmax,
                        ep_name,
                    )
            ok += 1
    else:
        for i in range(args.num_episodes):
            p = ds / f"episode_{i}.hdf5"
            if not p.is_file():
                print(f"episode_{i:03d}: SKIP (missing file)")
                continue
            with h5py.File(p, "r") as root:
                qpos = root["/observations/qpos"][()]
            raw_l = qpos[:, JOINTS_PER_ARM - 1].astype(np.float64)
            raw_r = qpos[:, 2 * JOINTS_PER_ARM - 1].astype(np.float64)
            u_l = remap_01(raw_l, lmin, lmax)
            u_r = remap_01(raw_r, rmin, rmax)
            b_l = binarize01(u_l)
            b_r = binarize01(u_r)
            n = len(raw_l)
            t = np.arange(n)

            n_open_l = int(np.sum(b_l == 0))
            n_close_l = int(np.sum(b_l == 1))
            n_open_r = int(np.sum(b_r == 0))
            n_close_r = int(np.sum(b_r == 1))

            print(
                f"episode_{i:03d}: n={n}  "
                f"L raw[{raw_l.min():.4f},{raw_l.max():.4f}] "
                f"u[{u_l.min():.3f},{u_l.max():.3f}] "
                f"open={n_open_l} close={n_close_l} trans={count_transitions(b_l)}  |  "
                f"R raw[{raw_r.min():.4f},{raw_r.max():.4f}] "
                f"u[{u_r.min():.3f},{u_r.max():.3f}] "
                f"open={n_open_r} close={n_close_r} trans={count_transitions(b_r)}"
            )

            if args.plots_dir is not None:
                if args.remap_01_only:
                    plot_episode_remap_01_only(
                        args.plots_dir / f"episode_{i:03d}.png",
                        t, raw_l, raw_r, u_l, u_r,
                        lmin, lmax, rmin, rmax, f"episode_{i:03d}",
                    )
                else:
                    plot_episode(
                        args.plots_dir / f"episode_{i:03d}.png",
                        t,
                        raw_l,
                        raw_r,
                        u_l,
                        u_r,
                        b_l,
                        b_r,
                        lmin,
                        lmax,
                        rmin,
                        rmax,
                        f"episode_{i:03d}",
                    )
            ok += 1

    print(f"\n共处理 {ok} 条 episode。")
    if args.plots_dir:
        mode = "remap [0,1] only (state)" if args.remap_01_only else "raw / remap / binary"
        print(f"图已写入: {args.plots_dir.resolve()}  ({mode})")


if __name__ == "__main__":
    main()
