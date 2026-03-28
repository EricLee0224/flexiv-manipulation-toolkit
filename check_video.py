import sys
import h5py
import cv2
import numpy as np

file_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/place_tube_0327/episode_000.hdf5"

STEP = 1
FPS = 30
SAVE_VIDEO = True

CAM_NAMES = ["left_cam0", "left_cam1", "right_cam0", "right_cam1"]


def decode(buf):
    if len(buf) == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return img


# --- pre-read all JPEG buffers into memory ---
print(f"Loading {file_path} ...")

with h5py.File(file_path, "r") as f:
    cam = f["camera"]
    n = len(cam["left_cam0"])
    print(f"Total frames: {n}")

    cam_data = {}
    for name in CAM_NAMES:
        print(f"  reading {name} ...")
        cam_data[name] = [cam[name][i] for i in range(n)]

print("All frames loaded into memory.\n")

# --- playback from memory ---
writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("/tmp/check.mp4", fourcc, FPS, (1280, 960))

for i in range(0, n, STEP):
    l0 = decode(cam_data["left_cam0"][i])
    l1 = decode(cam_data["left_cam1"][i])
    r0 = decode(cam_data["right_cam0"][i])
    r1 = decode(cam_data["right_cam1"][i])

    top = np.hstack((l0, l1))
    bottom = np.hstack((r0, r1))
    grid = np.vstack((top, bottom))

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
