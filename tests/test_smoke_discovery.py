from pathlib import Path

from lawextractor.data import collect_hidden_rotor, load_public
from lawextractor.scientist import discover


def test_discovery_smoke(tmp_path: Path) -> None:
    paths = collect_hidden_rotor(
        tmp_path / "data",
        train_episodes=4,
        val_episodes=2,
        test_episodes=2,
        steps=12,
    )
    report = discover(
        load_public(paths.public),
        tmp_path / "disc",
        latent_dims=(1, 2),
        epochs=2,
        seed=3,
    )
    assert report.winner.latent_dim in (1, 2)
    assert Path(report.winner.model_path).exists()
