#!/usr/bin/env python3
"""
Copy only episode.hdf5 + visualization mp4 from each episode_* folder into a clean tree.

Example:
  python scripts/package_episode_hdf5_mp4.py \\
    --src dataset/0401test \\
    --out dataset/place_tube_150

Default mp4: full_visualization.mp4 (same as build_dataset / recording layout).
If missing, uses the single *.mp4 in the folder (warns if several).

If many episodes have no visualization yet, use ``--mp4-optional`` to still
copy every ``episode.hdf5`` (mp4 copied only when present).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


EPISODE_DIR_RE = re.compile(r"^episode_(\d+)$")


def find_episode_dirs(src: Path) -> list[Path]:
    dirs = []
    for p in src.iterdir():
        if p.is_dir() and EPISODE_DIR_RE.match(p.name):
            dirs.append(p)
    return sorted(dirs, key=lambda d: int(EPISODE_DIR_RE.match(d.name).group(1)))


def pick_mp4(ep_dir: Path) -> Path | None:
    preferred = ep_dir / "full_visualization.mp4"
    if preferred.is_file():
        return preferred
    mp4s = sorted(ep_dir.glob("*.mp4"))
    if not mp4s:
        return None
    if len(mp4s) > 1:
        print(f"WARN {ep_dir.name}: multiple .mp4, using {mp4s[0].name}", file=sys.stderr)
    return mp4s[0]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Package episode.hdf5 + mp4 into a new folder")
    ap.add_argument(
        "--src",
        type=Path,
        default=root / "dataset" / "0401test",
        help="Source task dir containing episode_*/",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "dataset" / "place_tube_150",
        help="Output directory (created); each episode_* gets hdf5 + mp4 only",
    )
    ap.add_argument(
        "--sequential-names",
        action="store_true",
        help="Rename to episode_000, episode_001, ... in sort order (default: keep source names)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies only",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting non-empty --out",
    )
    ap.add_argument(
        "--mp4-optional",
        action="store_true",
        help="Copy episode.hdf5 even when no .mp4; copy mp4 only if it exists",
    )
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()

    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    episodes = find_episode_dirs(src)
    if not episodes:
        raise SystemExit(f"No episode_* directories under {src}")

    if out.exists():
        if not args.force and any(out.iterdir()):
            raise SystemExit(f"Output exists and is not empty: {out}  (use --force to overwrite)")
    elif not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_hdf5_only = 0
    for i, ep in enumerate(episodes):
        h5 = ep / "episode.hdf5"
        if not h5.is_file():
            print(f"SKIP {ep.name}: missing episode.hdf5", file=sys.stderr)
            continue
        mp4 = pick_mp4(ep)
        if mp4 is None and not args.mp4_optional:
            print(f"SKIP {ep.name}: no .mp4 (expected full_visualization.mp4)", file=sys.stderr)
            continue
        if mp4 is None and args.mp4_optional:
            n_hdf5_only += 1

        if args.sequential_names:
            dst_dir = out / f"episode_{i:03d}"
        else:
            dst_dir = out / ep.name

        if args.dry_run:
            extra = f"  {mp4.name}" if mp4 else "  (no mp4)"
            print(f"  {ep.name} -> {dst_dir.name}/  {h5.name}{extra}")
            n_ok += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(h5, dst_dir / "episode.hdf5")
        if mp4 is not None:
            shutil.copy2(mp4, dst_dir / mp4.name)
        n_ok += 1

    print(f"Done: {n_ok}/{len(episodes)} episodes -> {out}")
    if args.mp4_optional and n_hdf5_only:
        print(f"  ({n_hdf5_only} episodes had no .mp4; only episode.hdf5 was copied)")
    if args.dry_run:
        print("(dry-run; no files written)")


if __name__ == "__main__":
    main()
