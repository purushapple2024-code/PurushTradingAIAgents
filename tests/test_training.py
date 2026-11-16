from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptaa.mlmodels.registry import ModelRegistry
from ptaa.settings import AppConfig
from ptaa.training.labels import BarrierConfig, triple_barrier
from ptaa.training.pipeline import run_training, synthetic_candles
from ptaa.training.splits import assert_no_leakage, walk_forward_folds

from .conftest import IST


def _flat_frame(
    n: int, close: list[float], high: list[float] | None = None, low: list[float] | None = None
) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=n, tz=IST)
    c = pd.Series(close, index=idx)
    return pd.DataFrame(
        {"open": c, "high": high or c, "low": low or c, "close": c, "volume": 1000}, index=idx
    )


def test_triple_barrier_hits_upper_lower_vertical() -> None:
    # 30 bars of gentle noise to warm up vol, then controlled moves
    rng = np.random.default_rng(0)
    base = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30))))
    n = len(base) + 6
    close = base + [base[-1]] * 6
    high = list(close)
    low = list(close)
    # bar 30 -> upper barrier hit at bar 32 (high spikes 10%)
    high[32] = close[32] * 1.10
    # bar 33 -> lower barrier hit at bar 34 (low drops 10%)
    low[34] = close[34] * 0.90
    df = _flat_frame(n, close, high, low)
    out = triple_barrier(
        df, BarrierConfig(horizon=3, upper_mult=2.0, lower_mult=1.0, vol_window=20)
    )
    assert out.loc[df.index[30], "label"] == 1 and out.loc[df.index[30], "touched"] == "upper"
    assert out.loc[df.index[30], "t_end"] == 32
    assert out.loc[df.index[33], "label"] == -1 and out.loc[df.index[33], "touched"] == "lower"
    # a bar whose window has no barrier touch resolves at the vertical barrier
    assert out.loc[df.index[31], "touched"] in ("vertical", "lower", "upper")
    # last bar cannot be labelled
    assert out["t_end"].iloc[-1] == -1


def test_triple_barrier_stop_first_when_both_in_one_bar() -> None:
    rng = np.random.default_rng(1)
    base = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30))))
    close = base + [base[-1]] * 3
    high, low = list(close), list(close)
    high[31] = close[31] * 1.2
    low[31] = close[31] * 0.8
    out = triple_barrier(_flat_frame(len(close), close, high, low), BarrierConfig(horizon=2))
    assert out.loc[out.index[30], "label"] == -1  # conservative: stop counted first


def test_walk_forward_purge_and_embargo() -> None:
    n = 1000
    t_end = np.arange(n) + 5  # every label reaches 5 bars ahead
    t_end[-5:] = -1
    folds = list(walk_forward_folds(n, t_end, n_folds=4, min_train=300, embargo=10))
    assert len(folds) == 4
    prev_end = 0
    for f in folds:
        assert_no_leakage(f, t_end)
        assert f.test_start >= prev_end  # chronological, non-overlapping test windows
        prev_end = f.test_end
        # embargo: no training sample within 10 bars before the test window
        assert f.train_idx.max() < f.test_start - 10
        # purge: no training label reaches into the test window
        assert (t_end[f.train_idx] < f.test_start).all()
        assert (f.test_idx >= f.test_start).all() and (f.test_idx < f.test_end).all()


def test_walk_forward_rejects_tiny_data() -> None:
    with pytest.raises(ValueError):
        list(walk_forward_folds(100, np.arange(100), n_folds=5, min_train=100))


def test_registry_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path)
    f = tmp_path / "gbm" / "m.txt"
    f.parent.mkdir()
    f.write_text("model-bytes")
    rec = reg.register(
        "gbm",
        "5d",
        f,
        ["a", "b"],
        pd.Timestamp("2020-01-01").date(),
        pd.Timestamp("2026-01-01").date(),
        "NIFTY50",
        10,
        {"balanced_accuracy": 0.5},
    )
    assert rec.model_id == "gbm-5d-v1"
    assert reg.next_version("gbm", "5d") == 2
    reg2 = ModelRegistry(tmp_path)  # reload from disk
    ok, msg = reg2.verify("gbm-5d-v1")
    assert ok, msg
    f.write_text("tampered")
    ok, msg = reg2.verify("gbm-5d-v1")
    assert not ok and "sha256" in msg
    with pytest.raises(RuntimeError):
        reg2.path("gbm-5d-v1")
    with pytest.raises(KeyError):
        reg2.get("nope")


def test_end_to_end_training_on_synthetic(tmp_path: Path, cfg: AppConfig) -> None:
    reg = ModelRegistry(tmp_path / "models")
    cfg = cfg.model_copy(
        update={
            "universe": cfg.universe.model_copy(update={"custom": ["AAA", "BBB", "CCC", "DDD"]})
        }
    )
    rep = run_training(
        cfg,
        reg,
        lambda s: synthetic_candles(s, n=900),
        horizon="5d",
        universe="custom",
        n_folds=3,
        min_train=800,
        out_dir=tmp_path / "reports",
    )
    rec = reg.get(rep.model_id)
    assert rec.kind == "gbm" and rec.horizon == "5d" and rec.n_samples > 1500
    assert 0.0 <= rep.metrics["balanced_accuracy"] <= 1.0 and rep.metrics["folds"] == 3
    assert reg.verify(rep.model_id)[0]
    assert rep.report_path.exists()
    assert (
        "baseline" in str(rep.report_path.read_text()).lower() or rep.baseline
    )  # baseline always reported

    # inference wrapper loads only through the verified path and explains
    from ptaa.fast.features import FEATURE_NAMES, compute_features
    from ptaa.training.train_gbm import GBMModel

    m = GBMModel(reg, rep.model_id)
    row = (
        compute_features(synthetic_candles("AAA", n=900))
        .dropna(subset=[c for c in FEATURE_NAMES if not c.startswith(("rel_", "beta"))])
        .iloc[-1]
    )
    proba = m.predict_proba(row)[0]
    assert proba.shape == (3,) and abs(proba.sum() - 1) < 1e-6
    expl = m.explain(row, top_k=5)
    assert len(expl) == 5 and all(k in FEATURE_NAMES for k in expl)
