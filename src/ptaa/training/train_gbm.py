"""Walk-forward LightGBM trainer for the technical analyst.

Procedure: for each purged fold, fit on train, score test, collect out-of-fold probabilities and
metrics; then refit on all labelled data for the deployable model, register it with the averaged
out-of-fold metrics (never in-sample metrics), and save SHAP background statistics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ptaa.fast.features import FEATURE_NAMES
from ptaa.mlmodels.registry import ModelRecord, ModelRegistry
from ptaa.training.dataset import Dataset
from ptaa.training.metrics import classification_metrics
from ptaa.training.splits import assert_no_leakage, walk_forward_folds

log = logging.getLogger(__name__)
CLASSES = [-1, 0, 1]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "num_threads": 0,
}


@dataclass
class TrainResult:
    record: ModelRecord
    oof_proba: pd.DataFrame  # columns p_down, p_flat, p_up aligned to dataset rows that were tested
    fold_metrics: list[dict[str, float]] = field(default_factory=list)


def _to_class_idx(y: pd.Series) -> np.ndarray:
    return (y.to_numpy() + 1).astype(int)  # -1,0,1 -> 0,1,2


def _balanced_weights(y_idx: np.ndarray) -> np.ndarray:
    """Inverse class-frequency weights so the booster cannot win by predicting the majority."""
    counts = np.bincount(y_idx, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    w = (len(y_idx) / (3 * counts))[y_idx]
    return w / w.mean()


def train_gbm(
    ds: Dataset,
    registry: ModelRegistry,
    horizon: str,
    universe: str,
    params: dict[str, Any] | None = None,
    n_folds: int = 5,
    min_train: int = 2000,
    embargo: int = 10,
    num_boost_round: int = 600,
    early_stopping: int = 50,
) -> TrainResult:
    p = {**DEFAULT_PARAMS, **(params or {})}
    X = ds.X[FEATURE_NAMES]
    y_idx = _to_class_idx(ds.y)
    t_end = ds.meta["t_end"].to_numpy()
    n = len(ds)

    oof = np.full((n, 3), np.nan)
    fold_metrics: list[dict[str, float]] = []
    best_iters: list[int] = []

    for k, fold in enumerate(
        walk_forward_folds(n, t_end, n_folds=n_folds, min_train=min_train, embargo=embargo)
    ):
        assert_no_leakage(fold, t_end)
        tr, te = fold.train_idx, fold.test_idx
        if len(te) == 0 or len(tr) < 200:
            continue
        # inner validation = last 10% of the training window (chronological)
        cut = int(len(tr) * 0.9)
        w = _balanced_weights(y_idx[tr])
        dtr = lgb.Dataset(X.iloc[tr[:cut]], label=y_idx[tr[:cut]], weight=w[:cut])
        dva = lgb.Dataset(X.iloc[tr[cut:]], label=y_idx[tr[cut:]], weight=w[cut:])
        booster = lgb.train(
            p,
            dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        proba = booster.predict(X.iloc[te], num_iteration=booster.best_iteration)
        oof[te] = proba
        m = classification_metrics(ds.y.to_numpy()[te], proba, CLASSES)
        m["fold"] = float(k)
        m["n_train"], m["n_test"] = float(len(tr)), float(len(te))
        fold_metrics.append(m)
        best_iters.append(int(booster.best_iteration or num_boost_round))
        log.info(
            "fold %d: bal.acc %.3f top-decile %.3f (train %d, test %d)",
            k,
            m["balanced_accuracy"],
            m.get("top_decile_precision_up", float("nan")),
            len(tr),
            len(te),
        )

    if not fold_metrics:
        raise RuntimeError("no folds were evaluated; dataset too small")

    tested = ~np.isnan(oof[:, 0])
    agg = classification_metrics(ds.y.to_numpy()[tested], oof[tested], CLASSES)
    agg["folds"] = float(len(fold_metrics))
    agg["oof_rows"] = float(tested.sum())

    # Deployable model: refit on everything labelled with the median best iteration
    rounds = int(np.median(best_iters)) if best_iters else num_boost_round
    final = lgb.train(
        p,
        lgb.Dataset(X, label=y_idx, weight=_balanced_weights(y_idx)),
        num_boost_round=max(50, rounds),
    )

    model_dir = registry.root / "gbm"
    model_dir.mkdir(parents=True, exist_ok=True)
    version = registry.next_version("gbm", horizon)
    file = model_dir / f"gbm-{horizon}-v{version}.txt"
    final.save_model(str(file))

    # SHAP background: feature means/stds so inference can attribute without the training data
    bg = {"mean": X.mean().to_dict(), "std": X.std().to_dict()}
    (model_dir / f"gbm-{horizon}-v{version}.bg.json").write_text(json.dumps(bg), encoding="utf-8")

    rec = registry.register(
        kind="gbm",
        horizon=horizon,
        file=file.relative_to(registry.root),
        feature_names=FEATURE_NAMES,
        train_start=pd.Timestamp(ds.meta["ts"].min()).date(),
        train_end=pd.Timestamp(ds.meta["ts"].max()).date(),
        universe=universe,
        n_samples=n,
        metrics=agg,
        params={**p, "num_boost_round": rounds, "barrier_horizon": ds.horizon},
        notes="walk-forward purged CV; metrics are out-of-fold",
    )
    oof_df = pd.DataFrame(oof, columns=["p_down", "p_flat", "p_up"], index=ds.X.index)
    return TrainResult(record=rec, oof_proba=oof_df, fold_metrics=fold_metrics)


class GBMModel:
    """Inference wrapper: verified load, probabilities, SHAP top-k attributions."""

    def __init__(self, registry: ModelRegistry, model_id: str):
        self.record = registry.get(model_id)
        self.booster = lgb.Booster(model_file=str(registry.path(model_id)))
        self.features = self.record.feature_names

    def predict_proba(self, row: pd.Series | pd.DataFrame) -> np.ndarray:
        X = row.to_frame().T if isinstance(row, pd.Series) else row
        return self.booster.predict(X[self.features].astype(float))

    def explain(self, row: pd.Series, top_k: int = 8) -> dict[str, float]:
        """SHAP values for the 'up' class via LightGBM's native contribution output."""
        X = row.to_frame().T[self.features].astype(float)
        contrib = self.booster.predict(X, pred_contrib=True)[0]
        n = len(self.features)
        up = contrib[2 * (n + 1) : 3 * (n + 1) - 1]  # class index 2 = up; drop bias term
        s = pd.Series(up, index=self.features)
        s = s.reindex(s.abs().sort_values(ascending=False).index)[:top_k]
        return {k: round(float(v), 6) for k, v in s.items()}
