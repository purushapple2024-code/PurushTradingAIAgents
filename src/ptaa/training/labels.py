"""Triple-barrier labelling (Lopez de Prado).

For each bar t, look forward up to `horizon` bars. The label is decided by whichever barrier is
touched first: upper (profit target) -> +1, lower (stop) -> -1, vertical (time) -> sign of the
return at the horizon if |return| > `flat_mult` x vol, else 0 (flat). Barrier widths are
multiples of a
volatility estimate (ATR% or rolling std) so they scale with the instrument.

Also returns `t_end`: the bar index at which the label was decided. Purged cross-validation needs
it to know how far each training label "reaches" into the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarrierConfig:
    horizon: int = 5  # bars (vertical barrier)
    upper_mult: float = 2.0  # x volatility
    lower_mult: float = 1.0  # x volatility (stop closer than target: asymmetric, like the trader)
    vol_window: int = 20
    flat_mult: float = 0.25  # |ret| below flat_mult x vol at the horizon -> 0 (flat)


def rolling_vol(close: pd.Series, window: int) -> pd.Series:
    return np.log(close / close.shift(1)).rolling(window).std()


def triple_barrier(df: pd.DataFrame, cfg: BarrierConfig) -> pd.DataFrame:
    """Frame aligned to `df.index`: label (-1/0/1), ret, t_end (int position), touched."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = rolling_vol(df["close"], cfg.vol_window).to_numpy(dtype=float)
    n = len(df)
    label = np.zeros(n, dtype=np.int8)
    ret = np.full(n, np.nan)
    t_end = np.full(n, -1, dtype=np.int64)
    touched = np.array([""] * n, dtype=object)

    for i in range(n):
        if not np.isfinite(vol[i]) or vol[i] <= 0 or i + 1 >= n:
            continue
        entry = close[i]
        up = entry * (1 + cfg.upper_mult * vol[i])
        dn = entry * (1 - cfg.lower_mult * vol[i])
        last = min(n - 1, i + cfg.horizon)
        hit = None
        for j in range(i + 1, last + 1):
            # conservative: if both barriers are inside one bar, count the stop first
            if low[j] <= dn:
                hit, lab, r = j, -1, dn / entry - 1
                break
            if high[j] >= up:
                hit, lab, r = j, 1, up / entry - 1
                break
        if hit is None:
            hit = last
            r = close[last] / entry - 1
            lab = 0 if abs(r) <= cfg.flat_mult * vol[i] else (1 if r > 0 else -1)
            touched[i] = "vertical"
        else:
            touched[i] = "lower" if lab == -1 else "upper"
        label[i], ret[i], t_end[i] = lab, r, hit

    return pd.DataFrame(
        {"label": label, "ret": ret, "t_end": t_end, "touched": touched}, index=df.index
    )
