"""Dataset builder: candles for a universe -> stacked (features, labels, meta) frame.

Each symbol is processed independently (features and labels never cross symbols), then stacked
with a `symbol` column and sorted by time so walk-forward splits are chronological across the
whole panel. Relative-strength features use the NIFTY 50 index series when provided.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ptaa.fast.features import FEATURE_NAMES, compute_features
from ptaa.training.labels import BarrierConfig, triple_barrier

log = logging.getLogger(__name__)

CandleSource = Callable[[str], pd.DataFrame]  # symbol -> candles (IST index, OHLCV)


@dataclass
class Dataset:
    X: pd.DataFrame  # features, columns == FEATURE_NAMES
    y: pd.Series  # label in {-1, 0, 1}
    meta: pd.DataFrame  # symbol, ts, ret, t_end (global position), touched
    horizon: int

    def __len__(self) -> int:
        return len(self.X)


def build_dataset(
    symbols: list[str],
    candles: CandleSource,
    barrier: BarrierConfig,
    index_close: pd.Series | None = None,
    min_rows: int = 300,
) -> Dataset:
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        df = candles(sym)
        if df is None or len(df) < min_rows:
            log.warning("skip %s: %s rows", sym, 0 if df is None else len(df))
            continue
        feats = compute_features(df, index_close)
        labs = triple_barrier(df, barrier)
        both = feats.join(labs)
        both["symbol"] = sym
        both["ts"] = both.index
        both["local_pos"] = range(len(both))
        both = both[both["t_end"] >= 0]
        # keep rows where the core features exist (relative-strength may be NaN without an index)
        core = [c for c in FEATURE_NAMES if not c.startswith(("rel_strength", "beta"))]
        both = both.dropna(subset=core)
        parts.append(both)
    if not parts:
        raise ValueError("no symbols produced data")
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["ts", "symbol"], kind="stable").reset_index(drop=True)

    # Translate per-symbol t_end (local position) into a global "reach" position: the first panel
    # row whose timestamp is >= the label-decision timestamp. This keeps purging correct on a panel.
    decision_ts = pd.Series(index=panel.index, dtype="datetime64[ns, Asia/Kolkata]")
    for sym, grp in panel.groupby("symbol"):
        ts_by_local = pd.Series(grp["ts"].to_numpy(), index=grp["local_pos"].to_numpy())
        # t_end is a local position in the ORIGINAL candle frame, which includes dropped rows,
        # so map through the full timestamp index of that symbol's candles instead.
        full_ts = candles(sym).index
        decision_ts.loc[grp.index] = pd.DatetimeIndex(full_ts[grp["t_end"].to_numpy()]).tz_convert(
            "Asia/Kolkata"
        )
        del ts_by_local
    sorted_ts = panel["ts"].to_numpy()
    global_reach = pd.Index(sorted_ts).searchsorted(decision_ts.to_numpy(), side="left")
    global_reach = pd.Series(global_reach, index=panel.index).clip(upper=len(panel) - 1)

    meta = pd.DataFrame(
        {
            "symbol": panel["symbol"],
            "ts": panel["ts"],
            "ret": panel["ret"],
            "t_end": global_reach.astype(int),
            "touched": panel["touched"],
        }
    )
    X = panel[FEATURE_NAMES].astype(float)
    y = panel["label"].astype(int)
    return Dataset(X=X, y=y, meta=meta, horizon=barrier.horizon)
