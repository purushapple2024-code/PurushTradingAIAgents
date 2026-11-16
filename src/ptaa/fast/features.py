"""Deterministic feature engine. All numbers the models see are computed here, in pandas/NumPy.

`compute_features(df)` takes a candle frame (open, high, low, close, volume, IST index) and
returns a frame of features aligned to it. Every feature at row t uses only rows <= t.
`FEATURE_NAMES` is the stable ordered contract the model registry hashes.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def compute_features(df: pd.DataFrame, index_close: pd.Series | None = None) -> pd.DataFrame:
    """Return features for each row of `df`. Rows without enough history are NaN."""
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"].astype(float)
    f = pd.DataFrame(index=df.index)

    # Returns and volatility
    r1 = np.log(c / c.shift(1))
    f["ret_1"] = r1
    f["ret_5"] = np.log(c / c.shift(5))
    f["ret_20"] = np.log(c / c.shift(20))
    f["vol_20"] = r1.rolling(20).std()
    f["vol_60"] = r1.rolling(60).std()
    f["vol_ratio_20_60"] = f["vol_20"] / f["vol_60"]

    # Trend
    for n in (20, 50, 200):
        e = _ema(c, n)
        f[f"ema{n}_dist"] = c / e - 1
        f[f"ema{n}_slope"] = e.pct_change(5, fill_method=None)
    f["ema20_50_cross"] = _ema(c, 20) / _ema(c, 50) - 1
    macd = _ema(c, 12) - _ema(c, 26)
    sig = _ema(macd, 9)
    f["macd"] = macd / c
    f["macd_hist"] = (macd - sig) / c

    # Momentum / oscillators
    f["rsi14"] = _rsi(c, 14)
    f["rsi14_slope"] = f["rsi14"].diff(3)
    hh, ll = h.rolling(14).max(), lo.rolling(14).min()
    f["stoch_k"] = (c - ll) / (hh - ll).replace(0, np.nan) * 100
    f["roc_10"] = c.pct_change(10, fill_method=None)

    # Range / ATR
    atr = _atr(df, 14)
    f["atr14_pct"] = atr / c
    f["range_pct"] = (h - lo) / c
    f["close_pos_in_range"] = (c - lo) / (h - lo).replace(0, np.nan)
    f["gap_pct"] = df["open"] / c.shift(1) - 1

    # Bollinger
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - ma20) / (2 * sd20).replace(0, np.nan)
    f["bb_width"] = (4 * sd20) / ma20

    # Volume
    vma20 = v.rolling(20).mean()
    f["vol_zscore_20"] = (v - vma20) / v.rolling(20).std().replace(0, np.nan)
    f["vol_ratio_5_20"] = v.rolling(5).mean() / vma20.replace(0, np.nan)
    dollar = c * v
    f["turnover_20_log"] = np.log1p(dollar.rolling(20).mean())
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope_10"] = obv.diff(10) / vma20.replace(0, np.nan)

    # 52-week style position (252 daily bars; on intraday frames this is the lookback in bars)
    n52 = min(252, max(20, len(df)))
    f["pos_52w"] = (c - lo.rolling(n52).min()) / (
        h.rolling(n52).max() - lo.rolling(n52).min()
    ).replace(0, np.nan)
    f["dist_20d_high"] = c / h.rolling(20).max() - 1
    f["dist_20d_low"] = c / lo.rolling(20).min() - 1

    # Relative strength vs index when provided
    if index_close is not None:
        ic = index_close.reindex(df.index).ffill()
        ir5, ir20 = np.log(ic / ic.shift(5)), np.log(ic / ic.shift(20))
        f["rel_strength_5"] = f["ret_5"] - ir5
        f["rel_strength_20"] = f["ret_20"] - ir20
        f["beta_60"] = (
            r1.rolling(60).cov(np.log(ic / ic.shift(1)))
            / np.log(ic / ic.shift(1)).rolling(60).var()
        )
    else:
        f["rel_strength_5"] = np.nan
        f["rel_strength_20"] = np.nan
        f["beta_60"] = np.nan

    # Calendar
    idx = df.index
    f["dow"] = idx.dayofweek.astype(float)
    f["minute_of_day"] = (idx.hour * 60 + idx.minute).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)


FEATURE_NAMES: list[str] = list(
    compute_features(
        pd.DataFrame(
            {
                "open": [1.0] * 5,
                "high": [1.0] * 5,
                "low": [1.0] * 5,
                "close": [1.0] * 5,
                "volume": [1] * 5,
            },
            index=pd.date_range("2026-01-01 09:15", periods=5, freq="min", tz="Asia/Kolkata"),
        )
    ).columns
)


def feature_schema_hash() -> str:
    """Hash of the ordered feature list; stored with each trained model and checked at load."""
    return hashlib.sha256("|".join(FEATURE_NAMES).encode()).hexdigest()[:16]


def feature_row_hash(row: pd.Series) -> str:
    """Hash of one feature row; attached to a Signal so the Verifier can tie it to the data."""
    arr = np.nan_to_num(row[FEATURE_NAMES].to_numpy(dtype=float), nan=0.0)
    return hashlib.sha256(np.round(arr, 8).tobytes()).hexdigest()[:16]
