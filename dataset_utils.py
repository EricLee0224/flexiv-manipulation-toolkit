from pathlib import Path


DATASET_ROOT = Path("dataset")


def sanitize_task_name(task_name: str) -> str:
    task_name = task_name.strip().replace(" ", "_")
    if not task_name:
        raise ValueError("Task name cannot be empty")
    return task_name


def get_task_dir(task_name: str) -> Path:
    safe_name = sanitize_task_name(task_name)
    task_dir = DATASET_ROOT / safe_name
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def get_next_episode_dir(task_name: str) -> Path:
    """
    Return the next episode directory:
        dataset/<task_name>/episode_XXX/
    """
    task_dir = get_task_dir(task_name)

    existing = sorted(task_dir.glob("episode_*"))
    existing = [p for p in existing if p.is_dir()]

    if not existing:
        next_idx = 0
    else:
        indices = []
        for p in existing:
            try:
                idx = int(p.name.split("_")[-1])
                indices.append(idx)
            except ValueError:
                continue

        next_idx = max(indices) + 1 if indices else 0

    episode_dir = task_dir / f"episode_{next_idx:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    return episode_dir
