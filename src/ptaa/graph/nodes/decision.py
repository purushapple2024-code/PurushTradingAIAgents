"""Research manager (templated until the LLM layer), Trader, Risk Manager, PM, Executor.

In milestone 3 every narrative is a deterministic template; milestone 5 swaps in the local LLM for
the debate and report while keeping every number from these nodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ptaa.data.universe import sector_of
from ptaa.execution.broker import OrderRequest
from ptaa.graph.context import GraphContext, audited
from ptaa.graph.state import (
    DebateTurn,
    Decision,
    GraphState,
    RiskVerdict,
    Thesis,
    TradeProposal,
)
from ptaa.ledger.models import RunRow
from ptaa.risk import rules as risk_rules
from ptaa.risk.sizing import size_by_risk

HORIZON_MIN = {"swing": 5 * 375, "intraday": 90}  # 375 trading minutes per day
ATR_STOP = {"swing": 2.0, "intraday": 1.5}
ATR_TARGET = {"swing": 3.6, "intraday": 2.0}


@audited("research_manager")
def research_manager(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    ens = state["ensemble"]
    sigs = state.get("verified_signals", [])
    cited = [s.signal_id for s in sigs]
    risks = []
    if ens.regime in ("trend_down", "high_vol"):
        risks.append(f"regime {ens.regime}")
    if state.get("days_to_results") is not None and state["days_to_results"] <= 5:  # type: ignore[operator]
        risks.append(f"results in {state['days_to_results']} days")
    if state.get("circuit_today"):
        risks.append("circuit hit today")
    price, atr = state["price"], state["atr"]
    stop_mult = ATR_STOP[state["module"]]
    inval = (
        price - stop_mult * atr
        if ens.direction == "long"
        else (price + stop_mult * atr if ens.direction == "short" else None)
    )
    fallback = (
        f"{ens.direction} · ensemble {ens.score:.2f} · regime {ens.regime} · weights {ens.weights}"
    )

    llm = ctx.llm
    if llm is not None and llm.enabled and sigs:
        from ptaa.llm.reasoning import run_debate, write_narrative

        debate = run_debate(
            llm,
            state["symbol"],
            state["module"],
            ens.regime,
            ens.score,
            sigs,
            rounds=ctx.cfg.llm.debate_rounds,
        )
        narrative, llm_risks = write_narrative(
            llm, state["symbol"], ens.direction, ens.regime, debate, sigs, fallback
        )
        risks = risks + [r for r in llm_risks if r not in risks]
    else:
        bull = DebateTurn(
            side="bull",
            round=1,
            cited=cited,
            text="; ".join(
                f"[{s.signal_id}] {s.analyst} P(up)={s.p_up:.2f}"
                for s in sigs
                if s.p_up >= s.p_down
            )
            or "no supporting signals",
        )
        bear = DebateTurn(
            side="bear",
            round=1,
            cited=cited,
            text="; ".join(
                f"[{s.signal_id}] {s.analyst} P(down)={s.p_down:.2f}"
                for s in sigs
                if s.p_down > s.p_up
            )
            or "no opposing signals",
        )
        debate, narrative = [bull, bear], fallback

    thesis = Thesis(
        direction=ens.direction,
        conviction=ens.score,
        key_risks=risks,
        invalidation_price=inval,
        narrative=narrative,
    )
    return {"debate": debate, "thesis": thesis}


def _risk_fraction(
    state: GraphState, ctx: GraphContext, ps: Any, stop_frac: float
) -> tuple[float, str]:
    """PPO sizer when a registered policy beat fixed sizing out of sample; else fixed fraction 1.0."""
    rec = ctx.registry.latest("ppo", "sizer")
    if rec is None or rec.metrics.get("beats_fixed", 0.0) < 1.0:
        return 1.0, "fixed-fraction size"
    try:
        if rec.model_id not in ctx.models:
            from ptaa.mlmodels.ppo import PPOSizer

            ctx.models[rec.model_id] = PPOSizer(ctx.registry.path(rec.model_id))
        from ptaa.mlmodels.ppo import build_obs

        ens = state["ensemble"]
        seq = next((s for s in state.get("verified_signals", []) if s.analyst == "sequence"), None)
        unc = (seq.uncertainty or 1.2) / 100 if seq else 0.012
        runs = ctx.ledger.runs(50)
        hits = [1 if (r.decision in ("BUY", "SELL")) else 0 for r in runs]
        obs = build_obs(
            ens.score,
            unc,
            ens.regime,
            ps.equity / ps.peak_equity - 1,
            float(np.mean(hits)) if hits else 0.5,
            stop_frac,
            ps.equity / ctx.cfg.paper_book_inr - 1,
            len(runs),
        )
        frac = ctx.models[rec.model_id].risk_fraction(obs)
        return frac, f"PPO size {frac:.2f} ({rec.model_id})"
    except Exception as e:  # noqa: BLE001 - sizing must never fail the run
        return 1.0, f"fixed-fraction size (ppo unavailable: {type(e).__name__})"


@audited("trader")
def trader(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    th = state["thesis"]
    if th is None or th.direction == "no_trade":
        return {"proposal": None}
    module = state["module"]
    price, atr = state["price"], state["atr"]
    side = "BUY" if th.direction == "long" else "SELL"
    if side == "SELL" and module == "swing":
        return {"proposal": None}  # cash-equity delivery cannot be shorted; intraday shorts only
    stop = price - ATR_STOP[module] * atr if side == "BUY" else price + ATR_STOP[module] * atr
    target = price + ATR_TARGET[module] * atr if side == "BUY" else price - ATR_TARGET[module] * atr
    limits = ctx.cfg.risk[module]
    ps = ctx.ledger.portfolio_state({state["symbol"]: price})
    frac, sizer_note = _risk_fraction(state, ctx, ps, abs(price - stop) / price)
    qty = size_by_risk(ps.equity, price, stop, limits, risk_fraction=frac)
    prop = TradeProposal(
        symbol=state["symbol"],
        module=module,
        side=side,
        entry=round(price, 2),
        stop_loss=round(stop, 2),
        target=round(target, 2),
        quantity=qty,
        horizon_minutes=HORIZON_MIN[module],
        rationale=f"{ATR_STOP[module]}x ATR stop, {ATR_TARGET[module]}x ATR target, {sizer_note}",
    )
    return {"proposal": prop}


@audited("risk_manager")
def risk_manager(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    prop = state.get("proposal")
    if prop is None:
        return {"verdict": None}
    module = prop.module
    limits = ctx.cfg.risk[module]
    ps = ctx.ledger.portfolio_state({prop.symbol: prop.entry})
    rec = ctx.registry.latest("gbm", "5d" if module == "swing" else "1d") or ctx.registry.latest(
        "gbm", "5d"
    )
    age = ctx.registry.freshness_days(rec) if rec else 999
    p = risk_rules.Proposal(
        symbol=prop.symbol,
        sector=sector_of(prop.symbol),
        module=module,
        side=prop.side,
        entry=prop.entry,
        stop=prop.stop_loss,
        target=prop.target,
        quantity=prop.quantity,
        atr=state["atr"],
        avg_turnover_cr=state.get("avg_turnover_cr", 0.0),
        circuit_today=state.get("circuit_today", False),
        days_to_results=state.get("days_to_results"),
        ensemble_score=state["ensemble"].score,
        model_age_days=age,
    )
    v = risk_rules.evaluate(p, ps, limits)
    personas = ["(rules decided this verdict; LLM personas off)"]
    if ctx.llm is not None and ctx.llm.enabled and state.get("risk_loops", 0) == 0:
        from ptaa.llm.reasoning import persona_views

        th = state.get("thesis")
        personas = persona_views(
            ctx.llm,
            f"{prop.side} {prop.quantity} {prop.symbol} @ {prop.entry} stop {prop.stop_loss} target {prop.target}",
            "; ".join(f"{r.rule}: {r.outcome} ({r.message})" for r in v.rules),
            th.key_risks if th else [],
        )
    verdict = RiskVerdict(
        outcome=v.outcome,
        quantity=v.quantity,
        rules_triggered=v.triggered,
        rules=[r.model_dump() for r in v.rules],
        persona_views=personas,
    )
    out: dict[str, Any] = {
        "verdict": verdict,
        "risk_loops": state.get("risk_loops", 0) + (1 if v.outcome == "resize" else 0),
    }
    if v.outcome == "resize":
        out["proposal"] = prop.model_copy(update={"quantity": v.quantity})
    return out


def after_risk(state: GraphState) -> str:
    v = state.get("verdict")
    if v is None or v.outcome == "reject":
        return "portfolio_manager"
    if v.outcome == "resize" and state.get("risk_loops", 0) < 2:
        return "risk_manager"  # re-check the resized proposal once
    return "portfolio_manager"


@audited("portfolio_manager")
def portfolio_manager(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    th, prop, v, ens = (
        state.get("thesis"),
        state.get("proposal"),
        state.get("verdict"),
        state["ensemble"],
    )
    if th is None or prop is None or v is None or v.outcome == "reject" or v.quantity <= 0:
        why = (
            "no proposal"
            if prop is None
            else (f"rejected: {', '.join(v.rules_triggered)}" if v else "no verdict")
        )
        action, qty = "HOLD", 0
        report = f"HOLD {state['symbol']} · {why} · ensemble {ens.score:.2f} · regime {ens.regime}"
    else:
        action, qty = prop.side, v.quantity
        report = (
            f"{prop.side} {v.quantity} {prop.symbol} @ {prop.entry} · stop {prop.stop_loss} · target {prop.target} · "
            f"ensemble {ens.score:.2f} ({ens.regime}) · risk {v.outcome}"
            + (f" [{', '.join(v.rules_triggered)}]" if v.rules_triggered else "")
            + f" · risks: {', '.join(th.key_risks) or 'none flagged'}"
        )
    if ctx.llm is not None and ctx.llm.enabled and th is not None:
        from ptaa.llm.reasoning import allowed_numbers, write_report

        sigs = state.get("verified_signals", [])
        prose = write_report(
            ctx.llm,
            state["symbol"],
            f"{action} {qty}",
            th.narrative,
            (f"{v.outcome}; triggered: {v.rules_triggered}" if v else "no verdict"),
            (v.persona_views if v else []),
            report,
            allowed_numbers(sigs),
            {s.signal_id for s in sigs},
        )
        report = f"{report}\n\n{prose}"
    return {
        "decision": Decision(
            action=action, symbol=state["symbol"], quantity=qty, report=report, decided_at=ctx.now()
        )
    }  # type: ignore[arg-type]


@audited("executor")
def executor(state: GraphState, ctx: GraphContext) -> dict[str, Any]:
    d, prop, ens, v = (
        state["decision"],
        state.get("proposal"),
        state["ensemble"],
        state.get("verdict"),
    )
    ctx.ledger.record_run(
        RunRow(
            run_id=state["run_id"],
            symbol=state["symbol"],
            module=state["module"],
            as_of=state["as_of"],
            decision=d.action,
            quantity=d.quantity,
            ensemble_score=ens.score,
            regime=ens.regime,
            verdict=v.outcome if v else None,
            report=d.report,
            created_at=ctx.now(),
        )
    )
    if d.action == "HOLD" or prop is None:
        if ctx.alerts is not None and v is not None and v.outcome == "reject":
            ctx.alerts.reject(state["symbol"], v.rules_triggered)
        return {}
    if ctx.broker.is_live:
        return {"errors": ["executor: live broker refused in v1 (paper only)"]}
    req = OrderRequest(
        symbol=prop.symbol,
        module=prop.module,
        side=prop.side,
        quantity=d.quantity,
        stop_loss=prop.stop_loss,
        target=prop.target,
        strategy=prop.strategy,
        run_id=state["run_id"],
        reason="entry",
    )
    fill = ctx.broker.place(req, prop.entry, state["as_of"])
    if ctx.alerts is not None and fill:
        ctx.alerts.fill(fill.side, fill.quantity, fill.symbol, fill.price, fill.fees, prop.module)
    return {"audit": [], "errors": []} if fill else {"errors": ["executor: fill failed"]}
