from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .worlds import HiddenRotorWorld


@dataclass
class DatasetPaths:
    public: Path
    private: Path


def _action(rng: np.random.Generator, *, test: bool) -> float:
    # Train mostly on modest interventions; test includes stronger unseen kicks.
    if rng.random() < 0.72:
        return 0.0
    amp = 1.0 if test else 0.6
    return float(rng.choice([-amp, amp]))


def collect_hidden_rotor(
    out_dir: str | Path,
    *,
    seed: int = 0,
    train_episodes: int = 28,
    val_episodes: int = 8,
    test_episodes: int = 8,
    steps: int = 60,
) -> DatasetPaths:
    """Generate public observations and a separate private receipt file."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = HiddenRotorWorld(seed=seed)
    rng = np.random.default_rng(seed + 991)

    obs_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    next_rows: list[np.ndarray] = []
    split_rows: list[int] = []
    episode_rows: list[int] = []
    hidden_rows: list[np.ndarray] = []
    next_hidden_rows: list[np.ndarray] = []

    split_counts = [(0, train_episodes), (1, val_episodes), (2, test_episodes)]
    episode = 0
    for split, count in split_counts:
        for _ in range(count):
            obs = world.reset(seed=seed * 10000 + episode + 17)
            hidden = world.private_state()
            for _t in range(steps):
                u = _action(rng, test=(split == 2))
                nxt = world.step(u)
                nxt_hidden = world.private_state()
                obs_rows.append(obs)
                act_rows.append(np.array([u], dtype=np.float32))
                next_rows.append(nxt)
                split_rows.append(split)
                episode_rows.append(episode)
                hidden_rows.append(hidden)
                next_hidden_rows.append(nxt_hidden)
                obs = nxt
                hidden = nxt_hidden
            episode += 1

    public_path = out / "public.npz"
    private_path = out / "private_truth.npz"
    np.savez_compressed(
        public_path,
        obs=np.stack(obs_rows),
        action=np.stack(act_rows),
        next_obs=np.stack(next_rows),
        split=np.asarray(split_rows, dtype=np.int64),
        episode=np.asarray(episode_rows, dtype=np.int64),
    )
    np.savez_compressed(
        private_path,
        hidden=np.stack(hidden_rows),
        next_hidden=np.stack(next_hidden_rows),
        split=np.asarray(split_rows, dtype=np.int64),
        episode=np.asarray(episode_rows, dtype=np.int64),
    )
    return DatasetPaths(public=public_path, private=private_path)


def load_public(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {"obs", "action", "next_obs", "split", "episode"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"public dataset missing fields: {sorted(missing)}")
    return {k: data[k] for k in required}
