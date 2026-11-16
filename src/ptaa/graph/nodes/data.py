"""Data collector: builds the feature row and market context the analysts and rules need."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ptaa.fast.features import FEATURE_NAMES, compute_features, feature_row_hash
from ptaa.graph.context import GraphContext, audited
from ptaa.graph.state import GraphState


def daily_turnover_cr(candles: pd.DataFrame, days: int = 20) -> float:
    """Average daily traded value in INR crore, correct for both daily and intraday bars."""
    tv = (candles["close"] * candles["volume"]).astype(float)
    per_day = tv.groupby(pd.DatetimeIndex(tv.index).normalize()).sum()
    per_day = per_day[per_day > 0].tail(days)
    return float(per_day.mean() / 1e7) if len(per_day) else 0.0


def market_snapshot(candles: pd.DataFrame, index_close: pd.Series | None = None) -> dict[str, Any]:
    """Pure helper: latest feature row + price/ATR/turnover context from a candle frame."""
    feats = compute_features(candles, index_close)
    core = [c for c in FEATURE_NAMES if not c.startswith(("rel_strength", "beta"))]
    feats = feats.dropna(subset=core)
    if feats.empty:
        raise ValueError("not enough history to compute features")
    row = feats.iloc[-1]
    last = candles.loc[row.name]
    atr = float(row["atr14_pct"] * last["close"])
    turnover_cr = daily_turnover_cr(candles)
    idx_ret = []
    if index_close is not None and len(index_close) > 2:
        idx_ret = list(
            np.log(index_close / index_close.shift(1)).dropna().tail(60).to_numpy(dtype=float)
        )
    from ptaa.mlmodels.seq import last_window

    win = last_window(candles.loc[: row.name])
    return {
        "seq_window": win[0].tolist() if win is not None else None,
        "features": {k: (None if pd.isna(v) else float(v)) for k, v in row.items()},
        "feature_hash": feature_row_hash(row),
        "price": float(last["close"]),
        "atr": atr,
        "avg_turnover_cr": turnover_cr,
        "index_returns": idx_ret,
        "as_of": row.name.to_pydatetime(),
    }


@audited("data_collector")
def data_collector(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    """If the caller pre-filled features (fast path / tests) keep them; else nothing to do here.

    The CLI fills the snapshot from a candle source before invoking the graph so that the graph itself
    has no I/O dependency on yfinance or brokers.
    """
    if "features" not in state:
        return {"errors": ["data_collector: state has no features; build a market_snapshot first"]}
    nan_features = {k: 0.0 for k, v in state["features"].items() if v is None}
    if nan_features:
        return {"features": {**state["features"], **nan_features}}
    return {}
