"""Gates: Verifier, Market Regime, Ensemble + Strategy Selector. All code, no LLM."""

from __future__ import annotations

from typing import Any

import numpy as np

from ptaa.graph.context import GraphContext, audited
from ptaa.graph.state import EnsembleScore, Evidence, GraphState, Regime, Signal


def _model_ok(ctx: GraphContext, model_id: str | None) -> bool:
    if model_id is None:
        return False
    try:
        return ctx.registry.verify(model_id)[0]
    except KeyError:  # unregistered model id
        return False


def _quote_ok(ctx: GraphContext, url: str | None, quote: str | None) -> bool:
    """The quote must appear in the page as fetched (from cache); the model's word is not enough."""
    if not url or not quote:
        return False
    try:
        page = ctx.fetcher(url)
    except Exception:  # noqa: BLE001 - unreachable page = unverifiable evidence
        return False
    from ptaa.research.extract import extract_text, quote_present

    return quote_present(quote, extract_text(page.text))


@audited("verifier")
def verifier(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    kept: list[Signal] = []
    rejected: list[Evidence] = []
    for sig in state.get("signals", []):
        good: list[Evidence] = []
        for ev in sig.evidence:
            ok = True
            if ev.source_kind == "model":
                ok = _model_ok(ctx, ev.model_id) and ev.feature_hash == state["feature_hash"]
            elif ev.source_kind == "feature":
                ok = ev.feature_hash == state["feature_hash"]
            elif ev.source_kind in ("web", "filing"):
                ok = _quote_ok(ctx, ev.source_url, ev.quote)
            (good if ok else rejected).append(ev)
        if good:
            kept.append(sig.model_copy(update={"evidence": good}))
    return {"verified_signals": kept, "rejected": rejected}


def classify_regime(index_returns: list[float], vol_lookback: int = 20) -> Regime:
    """Lightweight regime classifier on index log returns (HMM upgrade is optional later)."""
    r = np.asarray(index_returns[-60:], dtype=float)
    if r.size < vol_lookback:
        return "range"
    vol = r[-vol_lookback:].std() * np.sqrt(252)
    trend = r[-vol_lookback:].sum()
    long_vol = r.std() * np.sqrt(252)
    if vol > max(0.28, 1.6 * long_vol):
        return "high_vol"
    if trend > 0.02:
        return "trend_up"
    if trend < -0.02:
        return "trend_down"
    return "range"


@audited("regime")
def regime(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    reg = classify_regime(state.get("index_returns", []))
    return {"ensemble": EnsembleScore(score=0.5, direction="no_trade", regime=reg, weights={})}


# Default analyst weights per regime. Learned weights (stacked logistic on validation folds) replace
# these once enough runs exist; until then they are explicit and visible.
DEFAULT_WEIGHTS: dict[Regime, dict[str, float]] = {
    "trend_up": {"technical": 0.5, "sequence": 0.2, "news": 0.2, "fundamental": 0.1},
    "trend_down": {"technical": 0.45, "sequence": 0.25, "news": 0.2, "fundamental": 0.1},
    "range": {"technical": 0.4, "sequence": 0.3, "news": 0.2, "fundamental": 0.1},
    "high_vol": {"technical": 0.35, "sequence": 0.2, "news": 0.35, "fundamental": 0.1},
}

REGIME_PENALTY: dict[Regime, float] = {
    "trend_up": 0.0,
    "range": 0.03,
    "trend_down": 0.08,
    "high_vol": 0.12,
}


@audited("ensemble")
def ensemble(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    sigs = state.get("verified_signals", [])
    reg: Regime = state["ensemble"].regime if state.get("ensemble") else "range"
    weights = DEFAULT_WEIGHTS[reg]
    present = {s.analyst: s for s in sigs}
    if not present:
        return {"ensemble": EnsembleScore(score=0.0, direction="no_trade", regime=reg, weights={})}
    w_sum = sum(weights[a] for a in present)
    used = {a: weights[a] / w_sum for a in present}
    p_up = sum(used[a] * s.p_up for a, s in present.items())
    p_down = sum(used[a] * s.p_down for a, s in present.items())
    score = float(np.clip(p_up - REGIME_PENALTY[reg], 0.0, 1.0))
    if p_up > p_down and score >= 0.5:
        direction = "long"
    elif p_down > p_up and p_down >= 0.5:
        direction = "short"
    else:
        direction = "no_trade"
    return {"ensemble": EnsembleScore(score=score, direction=direction, regime=reg, weights=used)}  # type: ignore[arg-type]
