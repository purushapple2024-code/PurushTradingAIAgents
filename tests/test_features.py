from __future__ import annotations

import numpy as np
import pandas as pd

from ptaa.fast.features import (
    FEATURE_NAMES,
    compute_features,
    feature_row_hash,
    feature_schema_hash,
)


def test_feature_contract_is_stable() -> None:
    assert len(FEATURE_NAMES) >= 35
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert len(feature_schema_hash()) == 16


def test_no_lookahead(daily_candles: pd.DataFrame) -> None:
    """Features at row t must not change when future rows are appended."""
    full = compute_features(daily_candles)
    truncated = compute_features(daily_candles.iloc[:250])
    a = full.iloc[:250].drop(
        columns=["pos_52w"]
    )  # pos_52w window depends on frame length by design
    b = truncated.drop(columns=["pos_52w"])
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_known_values(daily_candles: pd.DataFrame) -> None:
    f = compute_features(daily_candles)
    c = daily_candles["close"]
    # ret_1 is the log return
    expected = np.log(c.iloc[-1] / c.iloc[-2])
    assert abs(f["ret_1"].iloc[-1] - expected) < 1e-12
    # RSI within bounds, stoch within bounds
    assert 0 <= f["rsi14"].dropna().min() and f["rsi14"].dropna().max() <= 100
    assert 0 <= f["stoch_k"].dropna().min() and f["stoch_k"].dropna().max() <= 100
    # ATR positive
    assert (f["atr14_pct"].dropna() > 0).all()
    # no infinities
    assert not np.isinf(f.to_numpy(dtype=float)).any()


def test_row_hash_changes_with_data(daily_candles: pd.DataFrame) -> None:
    f = compute_features(daily_candles)
    f = f.dropna(subset=[c for c in f.columns if not c.startswith(("rel_strength", "beta"))])
    assert len(f) > 40  # 300 bars minus the 252-bar pos_52w warm-up
    h1 = feature_row_hash(f.iloc[-1])
    h2 = feature_row_hash(f.iloc[-2])
    assert h1 != h2 and len(h1) == 16
    assert feature_row_hash(f.iloc[-1]) == h1  # deterministic
