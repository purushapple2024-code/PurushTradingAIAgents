"""ML analysts. Numbers come from registered models and the feature engine; text is a template."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ptaa.fast.features import FEATURE_NAMES
from ptaa.graph.context import GraphContext, audited
from ptaa.graph.state import Evidence, GraphState, Signal


def _row(state: GraphState) -> pd.Series:
    return pd.Series(state["features"]).reindex(FEATURE_NAMES).astype(float)


def _technical_model(ctx: GraphContext, horizon: str) -> Any | None:
    rec = ctx.registry.latest("gbm", horizon)
    if rec is None:
        return None
    if rec.model_id not in ctx.models:
        from ptaa.training.train_gbm import GBMModel

        ctx.models[rec.model_id] = GBMModel(ctx.registry, rec.model_id)
    return ctx.models[rec.model_id]


@audited("technical_analyst")
def technical_analyst(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    horizon = "5d" if state["module"] == "swing" else "1d"
    model = (
        _technical_model(ctx, horizon) or _technical_model(ctx, "5d") or _technical_model(ctx, "1d")
    )
    if model is None:
        return {"errors": ["technical_analyst: no registered gbm model; run `ptaa train gbm`"]}
    row = _row(state)
    proba = model.predict_proba(row)[0]  # down, flat, up
    attribution = model.explain(row, top_k=8)
    sig = Signal(
        signal_id="S1",
        analyst="technical",
        horizon=model.record.horizon,
        p_up=float(proba[2]),
        p_down=float(proba[0]),
        evidence=[
            Evidence(
                claim=f"P(up)={proba[2]:.2f}, P(down)={proba[0]:.2f} over {model.record.horizon}",
                source_kind="model",
                model_id=model.record.model_id,
                feature_hash=state["feature_hash"],
                attribution=attribution,
                fetched_at=ctx.now(),
            )
        ],
        summary=f"{model.record.model_id}: top drivers "
        + ", ".join(f"{k} {v:+.3f}" for k, v in list(attribution.items())[:3]),
    )
    return {"signals": [sig]}


@audited("fundamental_analyst")
def fundamental_analyst(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    """Until the quality-GBM exists (milestone 8), a transparent ratio rule set on the feature row.

    Uses only price-derived long-horizon features present now; emits a mild, low-weight signal.
    """
    f = state["features"]
    pos52 = f.get("pos_52w")
    ema200 = f.get("ema200_dist")
    if pos52 is None or ema200 is None or np.isnan(pos52) or np.isnan(ema200):
        return {}
    # above the 200-day average and in the upper half of the 52-week range -> quality trend
    score = 0.5 + 0.25 * np.tanh(ema200 * 10) + 0.25 * (pos52 - 0.5)
    score = float(np.clip(score, 0.05, 0.95))
    sig = Signal(
        signal_id="S4",
        analyst="fundamental",
        horizon="5d",
        p_up=score,
        p_down=1 - score,
        evidence=[
            Evidence(
                claim=f"ema200_dist={ema200:+.3f}, pos_52w={pos52:.2f}",
                source_kind="feature",
                feature_hash=state["feature_hash"],
                attribution={"ema200_dist": float(ema200), "pos_52w": float(pos52)},
                fetched_at=ctx.now(),
            )
        ],
        summary="rule-based trend-quality proxy (quality-GBM lands in milestone 8)",
    )
    return {"signals": [sig]}


@audited("sequence_analyst")
def sequence_analyst(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    """LSTM (ONNX) expected move and uncertainty -> probabilities. Skipped when no model or no window."""
    rec = ctx.registry.latest(
        "seq_lstm", "5d" if state["module"] == "swing" else "1d"
    ) or ctx.registry.latest("seq_lstm", "5d")
    window = state.get("seq_window")
    if rec is None or window is None:
        return {}
    if rec.model_id not in ctx.models:
        from ptaa.mlmodels.seq import SeqModel

        ctx.models[rec.model_id] = SeqModel(
            ctx.registry.path(rec.model_id), int(rec.params.get("horizon_bars", 5))
        )
    model = ctx.models[rec.model_id]
    arr = np.asarray(window, dtype=np.float32)
    mean, std = model.predict(arr)
    p_up, p_down = model.to_probabilities(mean, std)
    sig = Signal(
        signal_id="S2",
        analyst="sequence",
        horizon=rec.horizon,
        p_up=p_up,
        p_down=p_down,
        expected_move_pct=round(mean * 100, 3),
        uncertainty=round(std * 100, 3),
        evidence=[
            Evidence(
                claim=f"expected {mean * 100:+.2f}% ± {std * 100:.2f}% over {rec.horizon}",
                source_kind="model",
                model_id=rec.model_id,
                feature_hash=state["feature_hash"],
                attribution={"mean_pct": round(mean * 100, 3), "std_pct": round(std * 100, 3)},
                fetched_at=ctx.now(),
            )
        ],
        summary=f"{rec.model_id}: {mean * 100:+.2f}% ± {std * 100:.2f}%",
    )
    return {"signals": [sig]}
