"""End-to-end training run: candles -> dataset -> walk-forward GBM -> registry -> backtest report.

`candle_source` is injectable so the same pipeline runs on yfinance (laptop/Colab), on the broker
history, or on synthetic data (tests, offline demo).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ptaa.data.universe import resolve_universe
from ptaa.mlmodels.registry import ModelRegistry
from ptaa.settings import AppConfig
from ptaa.training.backtest import backtest_signal, threshold_sweep
from ptaa.training.dataset import CandleSource, build_dataset
from ptaa.training.labels import BarrierConfig
from ptaa.training.train_gbm import train_gbm

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

HORIZON_BARS = {"1d": 1, "5d": 5, "1h": 1}


@dataclass
class RunReport:
    model_id: str
    metrics: dict[str, float]
    strategy: dict[str, float]
    baseline: dict[str, float]
    sweep: pd.DataFrame
    report_path: Path


def synthetic_candles(symbol: str, n: int = 1500, seed: int | None = None) -> pd.DataFrame:
    """Seeded GBM-with-regimes price path; different per symbol. For tests and offline demos."""
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) if seed is None else seed)
    idx = pd.bdate_range(end=datetime.now(tz=IST).date(), periods=n, tz=IST) + pd.Timedelta(
        hours=9, minutes=15
    )
    regime = np.sign(np.sin(np.linspace(0, 12, n)) + rng.normal(0, 0.3, n))
    ret = rng.normal(0.0003, 0.012, n) + 0.0015 * regime
    close = rng.uniform(300, 3000) * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(500_000, 8_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def yfinance_candle_source(cache_dir: Path, years: int = 12) -> CandleSource:
    from ptaa.data.cache import DataCache
    from ptaa.data.yfinance_adapter import YFinanceAdapter

    adapter = YFinanceAdapter(DataCache(cache_dir))
    end = datetime.now(tz=IST)
    start = end - timedelta(days=365 * years)

    def source(symbol: str) -> pd.DataFrame:
        return adapter.candles(symbol, "1d", start, end)

    return source


def run_training(
    cfg: AppConfig,
    registry: ModelRegistry,
    candle_source: CandleSource,
    horizon: str = "1d",
    universe: str | None = None,
    index_close: pd.Series | None = None,
    n_folds: int = 5,
    min_train: int = 2000,
    out_dir: Path | None = None,
    params: dict[str, Any] | None = None,
) -> RunReport:
    uni_name = universe or cfg.universe.swing
    symbols = resolve_universe(uni_name, cfg.universe.custom)
    barrier = BarrierConfig(horizon=HORIZON_BARS.get(horizon, 5), upper_mult=2.0, lower_mult=1.0)
    log.info("building dataset: %d symbols, horizon %s", len(symbols), horizon)
    ds = build_dataset(symbols, candle_source, barrier, index_close=index_close)
    log.info(
        "dataset: %d rows, label balance %s",
        len(ds),
        ds.y.value_counts(normalize=True).round(3).to_dict(),
    )

    result = train_gbm(
        ds,
        registry,
        horizon=horizon,
        universe=uni_name,
        n_folds=n_folds,
        min_train=min_train,
        params=params,
    )
    bt = backtest_signal(ds.meta, result.oof_proba, cfg.fees, index_close=index_close)
    sweep = threshold_sweep(ds.meta, result.oof_proba, cfg.fees)

    out_dir = out_dir or (registry.root / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": result.record.model_id,
        "trained_at": result.record.trained_at.isoformat(),
        "universe": uni_name,
        "rows": len(ds),
        "label_balance": ds.y.value_counts(normalize=True).round(4).to_dict(),
        "oof_metrics": result.record.metrics,
        "fold_metrics": result.fold_metrics,
        "backtest_strategy": bt.strategy,
        "backtest_baseline": bt.baseline,
        "threshold_sweep": sweep.to_dict(orient="records"),
        "disclaimer": (
            "Out-of-fold, fees and slippage included. Past performance is not indicative "
            "of future results. Not investment advice."
        ),
    }
    path = out_dir / f"{result.record.model_id}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return RunReport(
        result.record.model_id, result.record.metrics, bt.strategy, bt.baseline, sweep, path
    )
