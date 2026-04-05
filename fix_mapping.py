"""
Fix camera and gripper left/right mapping in existing episodes.

Old (wrong) mapping:
  bl_fisheye → left_cam0    (should be right_cam0)
  cl_fisheye → left_cam1    (should be left_cam0)
  br_fisheye → right_cam0   (should be right_cam1)
  cr_fisheye → right_cam1   (should be left_cam1)

  gripper2 → left   (should be right)
  gripper4 → right  (should be left)

This script swaps the pkl files and gripper data, then optionally
rebuilds the HDF5 episodes.

Usage:
    python fix_mapping.py dataset/place_tube_0327
    python fix_mapping.py dataset/place_tube_0327 --rebuild
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path


CAM_RENAME = {
    "left_cam0.pkl":  "right_cam0.pkl",   # bl → right_cam0
    "left_cam1.pkl":  "left_cam0.pkl",    # cl → left_cam0
    "right_cam0.pkl": "right_cam1.pkl",   # br → right_cam1
    "right_cam1.pkl": "left_cam1.pkl",    # cr → left_cam1
}


def fix_camera_pkls(episode_dir: Path):
    """Rename camera pkl files using temp names to avoid overwrite."""
    tmp_map = {}
    for old_name in CAM_RENAME:
        src = episode_dir / old_name
        if not src.exists():
            return False
        tmp = episode_dir / (old_name + ".tmp")
        shutil.move(str(src), str(tmp))
        tmp_map[tmp] = episode_dir / CAM_RENAME[old_name]

    for tmp, dst in tmp_map.items():
        shutil.move(str(tmp), str(dst))
    return True


def fix_gripper_pkl(episode_dir: Path):
    """Swap left/right in gripper.pkl."""
    path = episode_dir / "gripper.pkl"
    if not path.exists():
        return False

    with open(path, "rb") as f:
        data = pickle.load(f)

    swapped = {
        "left_timestamps":  data.get("right_timestamps", data.get("left_timestamps")),
        "left_pos":         data.get("right_pos", data.get("left_pos")),
        "right_timestamps": data.get("left_timestamps", data.get("right_timestamps")),
        "right_pos":        data.get("left_pos", data.get("right_pos")),
    }

    with open(path, "wb") as f:
        pickle.dump(swapped, f)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Task directory containing episode_* folders")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild HDF5 after fixing")
    args = parser.parse_args()

    target = Path(args.path)
    episodes = sorted(p for p in target.glob("episode_*") if p.is_dir())

    if not episodes:
        print(f"No episodes found in {target}")
        sys.exit(1)

    print(f"Fixing {len(episodes)} episodes in {target} ...")

    ok = 0
    for ep in episodes:
        cam_ok = fix_camera_pkls(ep)
        grip_ok = fix_gripper_pkl(ep)

        if cam_ok and grip_ok:
            ok += 1
        else:
            missing = []
            if not cam_ok:
                missing.append("camera pkls")
            if not grip_ok:
                missing.append("gripper.pkl")
            print(f"  SKIP {ep.name}: missing {', '.join(missing)}")

        # remove old HDF5 so it gets rebuilt fresh
        hdf5 = ep / "episode.hdf5"
        if hdf5.exists():
            hdf5.unlink()

    print(f"\nFixed: {ok}/{len(episodes)} episodes")

    if args.rebuild:
        print("\nRebuilding HDF5 ...")
        from build_dataset import build_episode
        built = 0
        for ep in episodes:
            if (ep / "arm.pkl").exists():
                print(f"\nBuilding: {ep.name}")
                if build_episode(ep):
                    built += 1
        print(f"\nRebuilt: {built}/{len(episodes)} episodes")


if __name__ == "__main__":
    main()
