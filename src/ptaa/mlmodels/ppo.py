"""PPO sizer inference (onnxruntime): a risk fraction in [0, 1]; rules cap the final quantity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ptaa.training.paper_env import OBS_DIM, REGIMES


def build_obs(
    score: float,
    uncertainty: float,
    regime: str,
    drawdown: float,
    hit_rate: float,
    stop_frac: float,
    equity_ret: float,
    n_trades: int,
) -> np.ndarray:
    reg = np.zeros(len(REGIMES), dtype=np.float32)
    reg[REGIMES.index(regime) if regime in REGIMES else 2] = 1.0
    core = np.array([score, uncertainty * 100, drawdown * 10, hit_rate], dtype=np.float32)
    tail = np.array([stop_frac * 100, equity_ret, n_trades / 1000.0], dtype=np.float32)
    return np.clip(np.concatenate([core, reg, tail]), -5, 5).astype(np.float32)[None, :]


class PPOSizer:
    def __init__(self, path: Path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        assert self.sess.get_inputs()[0].shape[-1] == OBS_DIM, "PPO policy obs dim mismatch"

    def risk_fraction(self, obs: np.ndarray) -> float:
        out = self.sess.run(None, {self.input_name: obs.astype(np.float32)})[0]
        return float(np.clip(out[0, 0], 0.0, 1.0))
