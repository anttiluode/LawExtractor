from pathlib import Path

from lawextractor.data import collect_hidden_rotor, load_public


def test_public_dataset_contains_no_hidden_state(tmp_path: Path) -> None:
    paths = collect_hidden_rotor(
        tmp_path,
        train_episodes=2,
        val_episodes=1,
        test_episodes=1,
        steps=5,
    )
    public = load_public(paths.public)
    assert "hidden" not in public
    assert "next_hidden" not in public
    assert paths.private.exists()
