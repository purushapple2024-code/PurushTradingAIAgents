"""Sequence model I/O shared by training (torch) and inference (onnxruntime).

Input: the last `window` bars as 6 channels: log return, high/close-1, low/close-1,
open/close-1, log volume z-score, intrabar range. Output: [mean_ret_h, log_std] for the horizon.
Training lives in training/train_seq.py (needs torch). Inference needs only onnxruntime and the
registry; uncertainty comes from the second head (log_std) learned with a Gaussian NLL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 120
CHANNELS = 6


def make_windows(df: pd.DataFrame, window: int = WINDOW) -> np.ndarray:
    """(n_windows, window, 6) float32 array; window i ends at bar i+window-1."""
    c = df["close"].to_numpy(dtype=float)
    o, h, lo = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low"))
    v = df["volume"].to_numpy(dtype=float)
    ret = np.diff(np.log(c), prepend=np.log(c[0]))
    lv = np.log1p(v)
    lv_z = (lv - pd.Series(lv).rolling(60, min_periods=10).mean().to_numpy()) / (
        pd.Series(lv).rolling(60, min_periods=10).std().to_numpy() + 1e-9
    )
    feats = np.stack(
        [ret, h / c - 1, lo / c - 1, o / c - 1, np.nan_to_num(lv_z), (h - lo) / c], axis=1
    ).astype(np.float32)
    n = len(df) - window + 1
    if n <= 0:
        return np.empty((0, window, CHANNELS), dtype=np.float32)
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    return feats[idx]


def last_window(df: pd.DataFrame, window: int = WINDOW) -> np.ndarray | None:
    """Last window computed with full history context (rolling volume z-score needs the lead-in)."""
    w = make_windows(df.tail(window + 60), window)
    return w[-1:] if len(w) else None


class SeqModel:
    """onnxruntime wrapper. Registry-verified path only."""

    def __init__(self, path: Path, horizon_bars: int):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.horizon_bars = horizon_bars

    def predict(self, window: np.ndarray) -> tuple[float, float]:
        """Return (expected log return over horizon, std)."""
        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        out = self.sess.run(None, {self.input_name: arr})[0]
        mean, log_std = float(out[0, 0]), float(out[0, 1])
        return mean, float(np.exp(np.clip(log_std, -8, 2)))

    @staticmethod
    def to_probabilities(mean: float, std: float, fee_hurdle: float = 0.003) -> tuple[float, float]:
        """P(return > hurdle), P(return < -hurdle) under a normal approximation."""
        from math import erf, sqrt

        def cdf(x: float) -> float:
            return 0.5 * (1 + erf(x / sqrt(2)))

        std = max(std, 1e-6)
        p_up = 1 - cdf((fee_hurdle - mean) / std)
        p_down = cdf((-fee_hurdle - mean) / std)
        return float(p_up), float(p_down)
