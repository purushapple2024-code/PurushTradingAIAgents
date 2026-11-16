"""PPO sizer trainer (Stable-Baselines3; Colab/Kaggle) with ONNX export of the deterministic policy.

Compares against fixed-fraction sizing on the held-out tail of the episode and registers the policy
only with that comparison recorded; the trader node still caps the result with the rules.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np

from ptaa.mlmodels.registry import ModelRegistry
from ptaa.settings import FeeConfig, RiskLimits
from ptaa.training.paper_env import OBS_DIM, Episode, PaperSizingEnv

log = logging.getLogger(__name__)


def _slice(ep: Episode, lo: int, hi: int) -> Episode:
    return Episode(
        *(
            getattr(ep, f)[lo:hi]
            for f in ("score", "uncertainty", "regime", "ret", "entry", "stop_frac")
        )
    )


def evaluate_policy(env: PaperSizingEnv, act: Any) -> dict[str, float]:
    obs, _ = env.reset()
    done, eq = False, []
    while not done:
        a = act(obs)
        obs, _, term, trunc, info = env.step(np.asarray([a], dtype=np.float32))
        eq.append(info["equity"])
        done = term or trunc
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    return {
        "final_equity": float(eq_arr[-1]),
        "max_drawdown": float((eq_arr / peak - 1).min()),
        "steps": float(len(eq)),
    }


def train_ppo(
    ep: Episode,
    limits: RiskLimits,
    fees: FeeConfig,
    registry: ModelRegistry,
    universe: str,
    timesteps: int = 200_000,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("train_ppo needs stable-baselines3 + torch; run on Colab/Kaggle") from e

    n = len(ep.score)
    cut = int(n * 0.8)
    train_env = PaperSizingEnv(_slice(ep, 0, cut), limits, fees)
    test_env = PaperSizingEnv(_slice(ep, cut, n), limits, fees)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=seed,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=timesteps)

    ppo_metrics = evaluate_policy(
        test_env, lambda o: float(model.predict(o, deterministic=True)[0][0])
    )
    fixed_metrics = evaluate_policy(PaperSizingEnv(_slice(ep, cut, n), limits, fees), lambda o: 1.0)
    half_metrics = evaluate_policy(PaperSizingEnv(_slice(ep, cut, n), limits, fees), lambda o: 0.5)

    # export deterministic actor to ONNX: obs -> action in [0,1]
    class Actor(torch.nn.Module):
        def __init__(self, policy: Any) -> None:
            super().__init__()
            self.policy = policy

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            feats = self.policy.extract_features(obs, self.policy.pi_features_extractor)
            latent = self.policy.mlp_extractor.forward_actor(feats)
            return torch.clamp(self.policy.action_net(latent), 0.0, 1.0)

    model_dir = registry.root / "ppo"
    model_dir.mkdir(parents=True, exist_ok=True)
    version = registry.next_version("ppo", "sizer")
    file = model_dir / f"ppo-sizer-v{version}.onnx"
    torch.onnx.export(
        Actor(model.policy).eval(),
        torch.zeros(1, OBS_DIM),
        str(file),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    metrics = (
        {f"ppo_{k}": v for k, v in ppo_metrics.items()}
        | {f"fixed_{k}": v for k, v in fixed_metrics.items()}
        | {f"half_{k}": v for k, v in half_metrics.items()}
    )
    metrics["beats_fixed"] = float(
        ppo_metrics["final_equity"] > fixed_metrics["final_equity"]
        and ppo_metrics["max_drawdown"] >= fixed_metrics["max_drawdown"]
    )
    rec = registry.register(
        kind="ppo",
        horizon="sizer",
        file=file.relative_to(registry.root),
        feature_names=[f"obs{i}" for i in range(OBS_DIM)],
        train_start=date(2000, 1, 1),
        train_end=date.today(),
        universe=universe,
        n_samples=n,
        metrics=metrics,
        params={"timesteps": timesteps, "seed": seed},
        notes="deploy only if beats_fixed == 1",
    )
    return {"model_id": rec.model_id, "metrics": metrics}
