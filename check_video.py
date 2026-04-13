import sys
import math
import h5py
import cv2
import numpy as np

file_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/place_tube_0327/episode_000.hdf5"

STEP = 1
FPS = 30
SAVE_VIDEO = True
CELL_W, CELL_H = 640, 480


def decode(buf):
    if len(buf) == 0:
        return np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    if (img.shape[1], img.shape[0]) != (CELL_W, CELL_H):
        img = cv2.resize(img, (CELL_W, CELL_H))
    return img


def make_grid(imgs, cam_names, ncols):
    """Arrange images into an ncols-wide grid, label each cell."""
    nrows = math.ceil(len(imgs) / ncols)
    blank = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    rows = []
    idx = 0
    for _ in range(nrows):
        cells = []
        for _ in range(ncols):
            if idx < len(imgs):
                cell = imgs[idx].copy()
                cv2.putText(cell, cam_names[idx], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
                cells.append(cell)
                idx += 1
            else:
                cells.append(blank)
        rows.append(np.hstack(cells))
    return np.vstack(rows)


# --- auto-detect cameras from HDF5 ---
print(f"Loading {file_path} ...")

with h5py.File(file_path, "r") as f:
    cam_group = f["camera"]
    cam_names = sorted(cam_group.keys())
    n = len(cam_group[cam_names[0]])
    print(f"Cameras: {cam_names}")
    print(f"Total frames: {n}")

    cam_data = {}
    for name in cam_names:
        print(f"  reading {name} ...")
        cam_data[name] = [cam_group[name][i] for i in range(n)]

print("All frames loaded into memory.\n")

ncols = 2 if len(cam_names) <= 4 else 3
nrows = math.ceil(len(cam_names) / ncols)
grid_w, grid_h = CELL_W * ncols, CELL_H * nrows

writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("/tmp/check.mp4", fourcc, FPS, (grid_w, grid_h))

for i in range(0, n, STEP):
    imgs = [decode(cam_data[name][i]) for name in cam_names]
    grid = make_grid(imgs, cam_names, ncols)

    try:
        cv2.imshow("dataset video", grid)
        key = cv2.waitKey(int(1000 / FPS))
        if key == 27:
            break
    except Exception:
        pass

    if writer is not None:
        writer.write(grid)

    if i % 100 == 0:
        print(f"frame {i}/{n}")

if writer is not None:
    writer.release()
    print("Saved video to /tmp/check.mp4")

cv2.destroyAllWindows()
