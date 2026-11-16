"""Gymnasium environment for the position sizer.

The agent does NOT pick direction or entries; those come from the pipeline (here: a stream of
out-of-fold signals with realised triple-barrier returns). At each candidate trade it observes
(ensemble score, model uncertainty, regime one-hot, drawdown, gross exposure, open positions,
recent hit-rate) and chooses a risk fraction in [0, 1] of the per-trade risk budget. The reward is
the fee-adjusted P&L of that trade scaled by equity, with a penalty on drawdown. Hard rule caps
still apply inside the env, so the agent learns to size *within* the rules, never around them.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ptaa.risk.sizing import size_by_risk
from ptaa.settings import FeeConfig, RiskLimits

REGIMES = ["trend_up", "trend_down", "range", "high_vol"]
OBS_DIM = 4 + len(REGIMES) + 3


@dataclass
class Episode:
    """One chronological stream of candidate trades."""

    score: np.ndarray  # ensemble score per candidate
    uncertainty: np.ndarray  # model std per candidate (fraction)
    regime: np.ndarray  # index into REGIMES
    ret: np.ndarray  # realised fractional return of the trade (after barriers), before fees
    entry: np.ndarray  # entry price
    stop_frac: np.ndarray  # |entry - stop| / entry


class PaperSizingEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(
        self,
        episode: Episode,
        limits: RiskLimits,
        fees: FeeConfig,
        equity: float = 1_000_000.0,
        fee_frac: float = 0.0032,
        max_steps: int | None = None,
    ):
        super().__init__()
        self.ep, self.limits, self.fees = episode, limits, fees
        self.start_equity, self.fee_frac = equity, fee_frac
        self.max_steps = max_steps or len(episode.score)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._i = 0
        self._equity = equity
        self._peak = equity
        self._hits: list[int] = []

    def _obs(self) -> np.ndarray:
        i = min(self._i, len(self.ep.score) - 1)
        reg = np.zeros(len(REGIMES), dtype=np.float32)
        reg[int(self.ep.regime[i])] = 1.0
        dd = self._equity / self._peak - 1.0
        hit = float(np.mean(self._hits[-20:])) if self._hits else 0.5
        core = np.array(
            [self.ep.score[i], self.ep.uncertainty[i] * 100, dd * 10, hit], dtype=np.float32
        )
        tail = np.array(
            [
                self.ep.stop_frac[i] * 100,
                self._equity / self.start_equity - 1.0,
                len(self._hits) / 1000.0,
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate([core, reg, tail]), -5, 5).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._i, self._equity, self._peak, self._hits = 0, self.start_equity, self.start_equity, []
        return self._obs(), {}

    def step(self, action: np.ndarray):  # type: ignore[override]
        i = self._i
        frac = float(np.clip(action[0], 0.0, 1.0))
        entry, stop_frac = float(self.ep.entry[i]), float(self.ep.stop_frac[i])
        qty = size_by_risk(
            self._equity, entry, entry * (1 - stop_frac), self.limits, risk_fraction=frac
        )
        notional = qty * entry
        pnl = notional * (float(self.ep.ret[i]) - self.fee_frac) if qty > 0 else 0.0
        self._equity += pnl
        self._peak = max(self._peak, self._equity)
        dd = self._peak - self._equity
        if qty > 0:
            self._hits.append(1 if pnl > 0 else 0)
        # reward: P&L in units of 1% equity, minus a drawdown penalty; halting rule as terminal
        reward = pnl / (0.01 * self.start_equity) - 0.1 * (dd / (0.01 * self.start_equity)) * (
            1 if pnl < 0 else 0
        )
        self._i += 1
        terminated = dd / self._peak >= self.limits.drawdown_halt_pct / 100
        truncated = self._i >= self.max_steps or self._i >= len(self.ep.score)
        info = {"equity": self._equity, "qty": qty, "pnl": pnl}
        return self._obs(), float(reward), bool(terminated), bool(truncated), info


def episode_from_backtest(
    meta: pd.DataFrame,
    oof_proba: pd.DataFrame,
    regimes: pd.Series | None = None,
    atr_frac: float = 0.012,
    stop_atr: float = 2.0,
) -> Episode:
    """Build an Episode from the GBM training pipeline outputs (meta + OOF probabilities)."""
    df = meta.join(oof_proba).dropna(subset=["p_up"]).sort_values("ts")
    df = df[df["p_up"] >= df[["p_down", "p_flat"]].max(axis=1)]
    n = len(df)
    reg = (
        regimes.reindex(df.index).fillna("range")
        if regimes is not None
        else pd.Series(["range"] * n, index=df.index)
    )
    return Episode(
        score=df["p_up"].to_numpy(dtype=np.float32),
        uncertainty=np.full(n, atr_frac, dtype=np.float32),
        regime=np.array([REGIMES.index(r) if r in REGIMES else 2 for r in reg], dtype=np.int64),
        ret=df["ret"].to_numpy(dtype=np.float32),
        entry=np.full(n, 1000.0, dtype=np.float32),
        stop_frac=np.full(n, atr_frac * stop_atr, dtype=np.float32),
    )
