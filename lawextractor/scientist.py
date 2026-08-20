from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LinearLaw(nn.Module):
    """The deliberately simple law family for Gate 0."""

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.A = nn.Linear(latent_dim, latent_dim, bias=True)
        self.B = nn.Linear(action_dim, latent_dim, bias=False)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.A(z) + self.B(u)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LatentScientist(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden: int = 48) -> None:
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim, hidden)
        self.decoder = Decoder(latent_dim, obs_dim, hidden)
        self.law = LinearLaw(latent_dim, action_dim)
        self.latent_dim = latent_dim

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z = self.encoder(obs)
        recon = self.decoder(z)
        z_next = self.law(z, action)
        pred_next = self.decoder(z_next)
        return z, recon, z_next, pred_next


@dataclass
class CandidateResult:
    latent_dim: int
    val_one_step_mse: float
    val_latent_consistency_mse: float
    law_parameters: int
    representation_parameters: int
    score: float
    model_path: str


@dataclass
class DiscoveryReport:
    candidates: list[CandidateResult]
    winner: CandidateResult

    def to_json(self) -> str:
        return json.dumps(
            {
                "candidates": [asdict(c) for c in self.candidates],
                "winner": asdict(self.winner),
            },
            indent=2,
        )


def _tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _standardize(train: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (all_x - mean) / std, mean, std


def prepare_arrays(public: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    train_mask = public["split"] == 0
    obs, obs_mean, obs_std = _standardize(public["obs"][train_mask], public["obs"])
    next_obs = (public["next_obs"] - obs_mean) / obs_std
    action, act_mean, act_std = _standardize(public["action"][train_mask], public["action"])
    return {
        **public,
        "obs": obs.astype(np.float32),
        "next_obs": next_obs.astype(np.float32),
        "action": action.astype(np.float32),
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std": obs_std.astype(np.float32),
        "act_mean": act_mean.astype(np.float32),
        "act_std": act_std.astype(np.float32),
    }


def fit_candidate(
    arrays: dict[str, np.ndarray],
    latent_dim: int,
    *,
    epochs: int = 450,
    lr: float = 2e-3,
    seed: int = 0,
    hidden: int = 48,
    device: str = "cpu",
    model_path: str | Path,
) -> CandidateResult:
    torch.manual_seed(seed + latent_dim * 101)
    np.random.seed(seed + latent_dim * 101)
    dev = torch.device(device)

    train = arrays["split"] == 0
    val = arrays["split"] == 1
    x = _tensor(arrays["obs"], dev)
    u = _tensor(arrays["action"], dev)
    y = _tensor(arrays["next_obs"], dev)

    model = LatentScientist(x.shape[1], u.shape[1], latent_dim, hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    train_idx = torch.as_tensor(np.where(train)[0], dtype=torch.long, device=dev)
    batch_size = min(512, len(train_idx))

    for _epoch in range(epochs):
        perm = train_idx[torch.randperm(len(train_idx), device=dev)]
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            z, recon, z_next, pred_next = model(x[idx], u[idx])
            with torch.no_grad():
                target_z_next = model.encoder(y[idx])

            recon_loss = torch.mean((recon - x[idx]) ** 2)
            pred_loss = torch.mean((pred_next - y[idx]) ** 2)
            consistency = torch.mean((z_next - target_z_next) ** 2)

            # Anti-collapse without prescribing any semantic axis. Each latent
            # coordinate must carry some variation across the training batch.
            centered = z - z.mean(dim=0, keepdim=True)
            variance = torch.mean(centered**2, dim=0)
            variance_floor = torch.mean(torch.relu(0.08 - variance))

            loss = recon_loss + 2.2 * pred_loss + 0.65 * consistency + 0.15 * variance_floor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    val_idx = torch.as_tensor(np.where(val)[0], dtype=torch.long, device=dev)
    model.eval()
    with torch.no_grad():
        z, _recon, z_next, pred_next = model(x[val_idx], u[val_idx])
        target_z = model.encoder(y[val_idx])
        val_pred = torch.mean((pred_next - y[val_idx]) ** 2).item()
        val_cons = torch.mean((z_next - target_z) ** 2).item()

    law_params = model.law.parameter_count
    rep_params = sum(p.numel() for p in model.encoder.parameters()) + sum(
        p.numel() for p in model.decoder.parameters()
    )

    # Gate 0's selection pressure: predictive fit first, then smaller ruler,
    # then smaller law. This is not claimed as a universal MDL objective.
    score = val_pred + 0.12 * val_cons + 0.0025 * latent_dim + 0.00002 * law_params

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": int(x.shape[1]),
            "action_dim": int(u.shape[1]),
            "latent_dim": latent_dim,
            "hidden": hidden,
            "obs_mean": arrays["obs_mean"],
            "obs_std": arrays["obs_std"],
            "act_mean": arrays["act_mean"],
            "act_std": arrays["act_std"],
        },
        path,
    )

    return CandidateResult(
        latent_dim=latent_dim,
        val_one_step_mse=val_pred,
        val_latent_consistency_mse=val_cons,
        law_parameters=law_params,
        representation_parameters=rep_params,
        score=score,
        model_path=str(path),
    )


def discover(
    public: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    latent_dims: Iterable[int] = (1, 2, 3, 4, 5),
    epochs: int = 450,
    seed: int = 0,
    device: str = "cpu",
) -> DiscoveryReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = prepare_arrays(public)
    results: list[CandidateResult] = []
    for k in latent_dims:
        results.append(
            fit_candidate(
                arrays,
                int(k),
                epochs=epochs,
                seed=seed,
                device=device,
                model_path=out / f"candidate_z{k}.pt",
            )
        )
    winner = min(results, key=lambda r: r.score)
    report = DiscoveryReport(candidates=results, winner=winner)
    (out / "discovery.json").write_text(report.to_json(), encoding="utf-8")
    return report


def load_model(path: str | Path, device: str = "cpu") -> tuple[LatentScientist, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LatentScientist(
        checkpoint["obs_dim"],
        checkpoint["action_dim"],
        checkpoint["latent_dim"],
        hidden=checkpoint["hidden"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
