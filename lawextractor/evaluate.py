from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from .scientist import load_model


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    eye = np.eye(xb.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    w = np.linalg.solve(xb.T @ xb + ridge * eye, xb.T @ y)
    return w[:-1], w[-1]


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2) + 1e-12
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def evaluate(
    public_path: str | Path,
    model_path: str | Path,
    *,
    private_path: str | Path | None = None,
    out_path: str | Path | None = None,
    device: str = "cpu",
) -> dict:
    public_npz = np.load(public_path)
    obs = public_npz["obs"].astype(np.float32)
    action = public_npz["action"].astype(np.float32)
    next_obs = public_npz["next_obs"].astype(np.float32)
    split = public_npz["split"]

    train, test = split == 0, split == 2

    # Baseline law lives directly in raw sensor coordinates.
    raw_x = np.concatenate([obs[train], action[train]], axis=1)
    raw_w, raw_b = _ridge_fit(raw_x, next_obs[train])
    raw_test_x = np.concatenate([obs[test], action[test]], axis=1)
    raw_pred = raw_test_x @ raw_w + raw_b
    raw_test_mse = _mse(next_obs[test], raw_pred)
    raw_law_params = int(raw_w.size + raw_b.size)

    model, ckpt = load_model(model_path, device=device)
    obs_z = (obs - ckpt["obs_mean"]) / ckpt["obs_std"]
    act_z = (action - ckpt["act_mean"]) / ckpt["act_std"]
    test_idx = np.where(test)[0]

    with torch.no_grad():
        x = torch.as_tensor(obs_z[test_idx], dtype=torch.float32, device=device)
        u = torch.as_tensor(act_z[test_idx], dtype=torch.float32, device=device)
        _z, _recon, _z_next, pred_next_std = model(x, u)
        pred_next = pred_next_std.cpu().numpy() * ckpt["obs_std"] + ckpt["obs_mean"]
        latent_test_mse = _mse(next_obs[test], pred_next)

    # A one-step fit can flatter a bad coordinate system. The harder receipt is
    # open-loop rollout: initialize both laws from the same observed state, then
    # feed the held-out interventions without correcting either model from reality.
    episode = public_npz["episode"]
    raw_roll_errors: list[float] = []
    latent_roll_errors: list[float] = []
    horizon = 30
    for ep in np.unique(episode[test]):
        idx = np.where((episode == ep) & test)[0][:horizon]
        if len(idx) == 0:
            continue
        raw_state = obs[idx[0]].copy()
        first_std = (raw_state - ckpt["obs_mean"][0]) / ckpt["obs_std"][0]
        with torch.no_grad():
            latent_state = model.encoder(
                torch.as_tensor(first_std[None], dtype=torch.float32, device=device)
            )
        for i in idx:
            raw_state = np.concatenate([raw_state, action[i]]) @ raw_w + raw_b
            action_std = (action[i] - ckpt["act_mean"][0]) / ckpt["act_std"][0]
            with torch.no_grad():
                latent_state = model.law(
                    latent_state,
                    torch.as_tensor(action_std[None], dtype=torch.float32, device=device),
                )
                latent_obs_std = model.decoder(latent_state).cpu().numpy()[0]
            latent_obs = latent_obs_std * ckpt["obs_std"][0] + ckpt["obs_mean"][0]
            truth = next_obs[i]
            raw_roll_errors.append(float(np.mean((raw_state - truth) ** 2)))
            latent_roll_errors.append(float(np.mean((latent_obs - truth) ** 2)))

    raw_rollout_mse = float(np.mean(raw_roll_errors))
    latent_rollout_mse = float(np.mean(latent_roll_errors))

    result = {
        "raw_sensor_law": {
            "test_one_step_mse": raw_test_mse,
            "open_loop_30_step_mse": raw_rollout_mse,
            "law_parameters": raw_law_params,
        },
        "invented_coordinate_law": {
            "latent_dim": int(ckpt["latent_dim"]),
            "test_one_step_mse": latent_test_mse,
            "open_loop_30_step_mse": latent_rollout_mse,
            "law_parameters": int(model.law.parameter_count),
            "one_step_improvement_factor": float(raw_test_mse / max(latent_test_mse, 1e-12)),
            "rollout_improvement_factor": float(raw_rollout_mse / max(latent_rollout_mse, 1e-12)),
        },
    }

    if private_path is not None:
        private = np.load(private_path)
        hidden = private["hidden"].astype(np.float32)
        with torch.no_grad():
            all_z = model.encoder(
                torch.as_tensor(obs_z, dtype=torch.float32, device=device)
            ).cpu().numpy()
        wz, bz = _ridge_fit(all_z[train], hidden[train])
        hidden_pred = all_z[test] @ wz + bz
        result["private_receipt_not_used_for_training"] = {
            "linear_r2_invented_z_to_true_hidden_state": _r2(hidden[test], hidden_pred)
        }

    if out_path is not None:
        Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
