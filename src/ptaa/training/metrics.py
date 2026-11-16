"""Classification and trading metrics for the walk-forward evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, classes: list[int]
) -> dict[str, float]:
    """`proba` columns follow `classes` (e.g. [-1, 0, 1])."""
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
    }
    if 1 in classes:
        p_up = proba[:, classes.index(1)]
        k = max(1, int(len(p_up) * 0.10))
        top = np.argsort(-p_up)[:k]
        out["top_decile_precision_up"] = float((y_true[top] == 1).mean())
        out["p_up_mean"] = float(p_up.mean())
    return out


def trading_metrics(ret: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Per-trade/per-bar return series -> annualised stats. Returns are net of fees."""
    r = ret.dropna()
    if r.empty:
        return {"n": 0.0}
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1).min()
    mean, std = r.mean(), r.std(ddof=1) if len(r) > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    return {
        "n": float(len(r)),
        "total_return": float(equity.iloc[-1] - 1),
        "sharpe": sharpe,
        "max_drawdown": float(dd),
        "win_rate": float((r > 0).mean()),
        "avg_return": float(mean),
    }
