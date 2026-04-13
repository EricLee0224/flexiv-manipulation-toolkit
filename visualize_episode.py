"""
Full-element episode visualization with 3D EE workspace view.

Layout (auto-scaled):
  Left  : camera grid (auto: 2×2 for 4 cams, 3+2 for 5 cams, etc.)
  Right-top : 3D EE trajectory + orientation frames
  Right-bot : EE orientation curves + gripper state

Usage:
    python visualize_episode.py dataset/test/episode_004
    python visualize_episode.py dataset/test/episode_004 --output my_video.mp4
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.spatial.transform import Rotation

# ---- layout constants (pixels) ----
DPI = 100
RIGHT_W = 960
VIEW3D_H = 420
STRIP_H = 300
RIGHT_H = VIEW3D_H + STRIP_H   # 720

TRAIL_LEN = 100
AXIS_LEN = 0.04

LEFT_BGR  = (60, 60, 230)
RIGHT_BGR = (230, 140, 40)

CAM_LABELS = {
    "left_cam0": "CL Fisheye",  "left_cam1": "CR Fisheye",
    "right_cam0": "BL Fisheye", "right_cam1": "BR Fisheye",
    "top_cam": "Top Cam (D435i)",
}


def _cam_label(name: str) -> str:
    return CAM_LABELS.get(name, name)


# =====================================================================
# helpers
# =====================================================================

def quat_wxyz_to_euler(q):
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)

def quat_wxyz_to_rotmat(q):
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()

def decode_jpeg(buf, w, h):
    if len(buf) == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if (img.shape[1], img.shape[0]) != (w, h):
        img = cv2.resize(img, (w, h))
    return img


def compute_cam_grid_layout(n_cams: int, panel_w: int, panel_h: int,
                             src_ratio: float = 4.0 / 3.0):
    """Compute grid dimensions and per-cell size, preserving source aspect ratio."""
    if n_cams <= 4:
        ncols, nrows = 2, 2
    elif n_cams <= 6:
        ncols, nrows = 3, 2
    elif n_cams <= 9:
        ncols, nrows = 3, 3
    else:
        ncols = math.ceil(math.sqrt(n_cams))
        nrows = math.ceil(n_cams / ncols)
    max_cw = panel_w // ncols
    max_ch = panel_h // nrows
    if max_cw / max_ch > src_ratio:
        cell_h = max_ch
        cell_w = int(cell_h * src_ratio)
    else:
        cell_w = max_cw
        cell_h = int(cell_w / src_ratio)
    return ncols, nrows, cell_w, cell_h


def build_cam_grid(cam_frames, cam_names, ncols, nrows, cell_w, cell_h):
    """Decode JPEG buffers and assemble into a grid image."""
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    imgs = []
    for j, name in enumerate(cam_names):
        img = decode_jpeg(cam_frames[name], cell_w, cell_h)
        cv2.putText(img, _cam_label(name), (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1, cv2.LINE_AA)
        imgs.append(img)
    while len(imgs) < ncols * nrows:
        imgs.append(blank)
    rows = []
    for r in range(nrows):
        rows.append(np.hstack(imgs[r * ncols:(r + 1) * ncols]))
    return np.vstack(rows)


# =====================================================================
# pre-render: 3D workspace
# =====================================================================

def render_3d_base(left_pos, right_pos):
    fig = plt.figure(figsize=(RIGHT_W / DPI, VIEW3D_H / DPI), dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0)

    ax.plot(*left_pos.T,  color="#e74c3c", alpha=0.12, lw=0.8)
    ax.plot(*right_pos.T, color="#3498db", alpha=0.12, lw=0.8)

    ax.scatter(*left_pos[0],  color="#e74c3c", s=25, marker="s", alpha=0.5, zorder=5)
    ax.scatter(*right_pos[0], color="#3498db", s=25, marker="s", alpha=0.5, zorder=5)

    ax.set_xlabel("X (m)", fontsize=7, labelpad=1)
    ax.set_ylabel("Y (m)", fontsize=7, labelpad=1)
    ax.set_zlabel("Z (m)", fontsize=7, labelpad=1)
    ax.set_title("Left (red)  /  Right (blue)  EE Workspace", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=28, azim=-55)

    all_pos = np.vstack([left_pos, right_pos])
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        lo, hi = all_pos[:, i].min(), all_pos[:, i].max()
        margin = max(0.02, (hi - lo) * 0.12)
        setter(lo - margin, hi + margin)

    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    base = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    img_h, img_w = base.shape[:2]
    fig_h = fig.canvas.get_width_height()[1]

    _proj = ax.get_proj()

    def _raw_project(pt):
        x2, y2, _ = proj3d.proj_transform(pt[0], pt[1], pt[2], _proj)
        px, py = ax.transData.transform((x2, y2))
        return px, fig_h - py

    need_resize = (img_w, img_h) != (RIGHT_W, VIEW3D_H)
    if need_resize:
        sx, sy = RIGHT_W / img_w, VIEW3D_H / img_h
        base = cv2.resize(base, (RIGHT_W, VIEW3D_H))

    def project(pt):
        px, py = _raw_project(pt)
        if need_resize:
            px, py = px * sx, py * sy
        return int(round(px)), int(round(py))

    return base, project, fig


# =====================================================================
# pre-render: bottom strip (orientation curves + gripper)
# =====================================================================

def render_strip_base(ts_sec, left_euler, right_euler, gripper_l, gripper_r):
    fig, axes = plt.subplots(
        2, 1, figsize=(RIGHT_W / DPI, STRIP_H / DPI), dpi=DPI)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.13, hspace=0.45)

    c3 = ["#e74c3c", "#2ecc71", "#3498db"]

    ax = axes[0]
    for k, (lbl, c) in enumerate(zip(["roll", "pitch", "yaw"], c3)):
        ax.plot(ts_sec, left_euler[:, k],  color=c, lw=0.7, label=f"L_{lbl}")
        ax.plot(ts_sec, right_euler[:, k], color=c, lw=0.7, ls="--", label=f"R_{lbl}")
    ax.set_ylabel("deg", fontsize=7)
    ax.set_title("EE Orientation (Euler XYZ)", fontsize=8, fontweight="bold")
    ax.legend(ncol=6, fontsize=5, loc="upper right", framealpha=0.6)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(ts_sec[0], ts_sec[-1])

    ax = axes[1]
    ax.plot(ts_sec, gripper_l,  color="#e74c3c", lw=1.0, label="Left")
    ax.plot(ts_sec, gripper_r,  color="#3498db", lw=1.0, label="Right")
    ax.set_ylabel("pos", fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_title("Gripper State", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(ts_sec[0], ts_sec[-1])
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    base = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    img_h, img_w = base.shape[:2]
    axes_px = []
    for a in axes:
        bb = a.get_position()
        axes_px.append((
            int(bb.x0 * img_w), int(bb.x1 * img_w),
            int((1 - bb.y1) * img_h), int((1 - bb.y0) * img_h),
        ))

    t_min, t_max = float(ts_sec[0]), float(ts_sec[-1])
    plt.close(fig)

    if (img_w, img_h) != (RIGHT_W, STRIP_H):
        sx, sy = RIGHT_W / img_w, STRIP_H / img_h
        base = cv2.resize(base, (RIGHT_W, STRIP_H))
        axes_px = [(int(x0*sx), int(x1*sx), int(y0*sy), int(y1*sy))
                    for (x0, x1, y0, y1) in axes_px]

    return base, axes_px, t_min, t_max


# =====================================================================
# per-frame 3D drawing
# =====================================================================

def draw_3d_overlay(base, project, i, left_pos, right_pos, left_rot, right_rot):
    frame = base.copy()
    trail_start = max(0, i - TRAIL_LEN)

    for pos_arr, rot_arr, color, label in [
        (left_pos,  left_rot,  LEFT_BGR,  "Left"),
        (right_pos, right_rot, RIGHT_BGR, "Right"),
    ]:
        pts = np.array([project(pos_arr[j]) for j in range(trail_start, i + 1)], np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

        c = project(pos_arr[i])
        for col, ac in enumerate([(0, 0, 255), (0, 200, 0), (255, 80, 0)]):
            end = pos_arr[i] + rot_arr[i][:, col] * AXIS_LEN
            cv2.arrowedLine(frame, c, project(end), ac, 2, cv2.LINE_AA, tipLength=0.25)

        cv2.circle(frame, c, 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, c, 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (c[0] + 10, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (c[0] + 10, c[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return frame


# =====================================================================
# main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--output", default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    ep = Path(args.episode_dir)
    hdf5 = ep / "episode.hdf5"
    if not hdf5.exists():
        print(f"episode.hdf5 not found in {ep}"); sys.exit(1)

    out_path = args.output or str(ep / "full_visualization.mp4")
    print(f"Loading {hdf5} ...")

    with h5py.File(hdf5, "r") as f:
        ts      = f["timestamp"][:]
        n       = len(ts)
        ts_sec  = (ts - ts[0]) / 1e9
        l_tcp   = f["left_arm/tcp_pose"][:]
        r_tcp   = f["right_arm/tcp_pose"][:]
        grip_l  = f["left_arm/gripper"][:]
        grip_r  = f["right_arm/gripper"][:]

        cam_names = sorted(f["camera"].keys())

    print(f"Cameras detected: {cam_names}")

    # compute left-panel layout
    ncols, nrows, cam_cw, cam_ch = compute_cam_grid_layout(
        len(cam_names), panel_w=RIGHT_W, panel_h=RIGHT_H)
    left_w = cam_cw * ncols
    left_h = cam_ch * nrows

    video_w = left_w + RIGHT_W
    video_h = max(left_h, RIGHT_H)

    l_pos, r_pos   = l_tcp[:, :3], r_tcp[:, :3]
    l_rot, r_rot   = quat_wxyz_to_rotmat(l_tcp[:, 3:]), quat_wxyz_to_rotmat(r_tcp[:, 3:])
    l_euler        = quat_wxyz_to_euler(l_tcp[:, 3:])
    r_euler        = quat_wxyz_to_euler(r_tcp[:, 3:])

    print("Rendering 3D base ...")
    base_3d, project, _fig = render_3d_base(l_pos, r_pos)

    print("Rendering plot strip ...")
    strip_base, strip_axes, t_min, t_max = render_strip_base(
        ts_sec, l_euler, r_euler, grip_l, grip_r)
    t_range = max(t_max - t_min, 1e-6)

    print("Loading camera frames ...")
    with h5py.File(hdf5, "r") as f:
        cam_data = {name: [f[f"camera/{name}"][i] for i in range(n)]
                    for name in cam_names}

    duration = ts_sec[-1]
    n_video = int(np.ceil(duration * args.fps))
    video_t = np.linspace(0, duration, n_video, endpoint=False)
    data_idx = np.searchsorted(ts_sec, video_t, side="right") - 1
    data_idx = np.clip(data_idx, 0, n - 1)

    actual_hz = (n - 1) / duration if duration > 0 else 0
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (video_w, video_h))
    print(f"Generating video: {n_video} video frames ({args.fps}fps), "
          f"{duration:.1f}s real-time, data {n} frames ({actual_hz:.1f}Hz), "
          f"{len(cam_names)} cameras, {video_w}x{video_h} → {out_path}")

    prev_di = -1
    cached_cam_grid = None
    cached_view3d = None

    for vi in range(n_video):
        di = data_idx[vi]
        t_now = video_t[vi]

        if di != prev_di:
            cam_frame_dict = {name: cam_data[name][di] for name in cam_names}
            cached_cam_grid = build_cam_grid(
                cam_frame_dict, cam_names, ncols, nrows, cam_cw, cam_ch)
            cached_view3d = draw_3d_overlay(base_3d, project, di,
                                            l_pos, r_pos, l_rot, r_rot)
            prev_di = di

        # strip with cursor
        strip = strip_base.copy()
        frac = (t_now - t_min) / t_range
        for (x0, x1, y0, y1) in strip_axes:
            xp = int(x0 + frac * (x1 - x0))
            cv2.line(strip, (xp, y0), (xp, y1), (0, 0, 255), 2)

        right = np.vstack([cached_view3d, strip])

        # pad left/right panels to the same height
        if cached_cam_grid.shape[0] < video_h:
            pad = np.zeros((video_h - cached_cam_grid.shape[0], left_w, 3), dtype=np.uint8)
            left_panel = np.vstack([cached_cam_grid, pad])
        else:
            left_panel = cached_cam_grid[:video_h]

        if right.shape[0] < video_h:
            pad = np.zeros((video_h - right.shape[0], RIGHT_W, 3), dtype=np.uint8)
            right = np.vstack([right, pad])
        else:
            right = right[:video_h]

        frame = np.hstack([left_panel, right])

        cv2.putText(frame, f"data #{di}/{n}  t={t_now:.2f}s", (10, video_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

        if vi % 300 == 0:
            print(f"  {vi}/{n_video} ({100*vi/n_video:.0f}%)")

    writer.release()
    plt.close(_fig)
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
