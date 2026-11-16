from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from ptaa.settings import AppConfig, load_config

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    from pathlib import Path

    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


@pytest.fixture
def daily_candles() -> pd.DataFrame:
    """300 synthetic daily bars with a mild uptrend and realistic noise. Seeded, deterministic."""
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.bdate_range("2025-01-01", periods=n, tz=IST) + pd.Timedelta(hours=9, minutes=15)
    ret = rng.normal(0.0004, 0.012, n)
    close = 1000 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


@pytest.fixture
def session_start() -> datetime:
    return datetime(2026, 11, 16, 9, 15, tzinfo=IST)


def ts_at(base: datetime, seconds: float) -> datetime:
    return base + timedelta(seconds=seconds)
