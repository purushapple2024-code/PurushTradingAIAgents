"""Offline evals: frozen snapshots run through the graph with every invariant asserted.

Invariants (each a named check, reported pass/fail):
  no_uncited_numbers   every number in LLM prose exists in the verified signals / proposal / verdict
  conviction_is_ensemble thesis.conviction == ensemble.score
  rules_never_violated  any BUY/SELL quantity passes evaluate() again with the final state
  deterministic         two runs on the same snapshot give the same decision and quantity
  provenance            every verified evidence has model_id+feature_hash or url+quote
  fast_path_latency     p99 tick->exit under the budget on a replayed tick stream
Snapshots are JSON files in evals/snapshots/; `ptaa eval --record N` creates them from synthetic
data or from cached real candles.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ptaa.graph.context import GraphContext
from ptaa.graph.graph import build_graph, run_once
from ptaa.graph.nodes.data import market_snapshot
from ptaa.llm.reasoning import _NUM, _canon, allowed_numbers
from ptaa.models import Tick

IST = ZoneInfo("Asia/Kolkata")
LATENCY_BUDGET_US = 5000.0


@dataclass
class EvalResult:
    snapshot: str
    checks: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


def record_snapshots(
    out_dir: Path, symbols: list[str], candle_source: Any, n: int = 30
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, sym in enumerate(symbols[:n]):
        df = candle_source(sym)
        snap = market_snapshot(df)
        snap["as_of"] = snap["as_of"].isoformat()
        p = out_dir / f"{i:03d}_{sym}.json"
        p.write_text(
            json.dumps({"symbol": sym, "module": "swing", "snapshot": snap}, default=str),
            encoding="utf-8",
        )
        paths.append(p)
    return paths


def _load(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    d["snapshot"]["as_of"] = datetime.fromisoformat(d["snapshot"]["as_of"])
    return d


def run_eval(ctx: GraphContext, snapshot_path: Path) -> EvalResult:
    from ptaa.data.universe import sector_of
    from ptaa.risk import rules as risk_rules

    d = _load(snapshot_path)
    res = EvalResult(snapshot=snapshot_path.name)
    graph = build_graph(ctx)
    out1 = run_once(graph, ctx, d["symbol"], d["module"], dict(d["snapshot"]))
    # undo any paper fill so the second run sees the same portfolio
    if out1.get("decision") and out1["decision"].action != "HOLD":
        ctx.broker.close_position(
            d["symbol"], d["snapshot"]["price"], d["snapshot"]["as_of"], "manual"
        )
    out2 = run_once(graph, ctx, d["symbol"], d["module"], dict(d["snapshot"]))
    if out2.get("decision") and out2["decision"].action != "HOLD":
        ctx.broker.close_position(
            d["symbol"], d["snapshot"]["price"], d["snapshot"]["as_of"], "manual"
        )

    res.checks["no_errors"] = out1.get("errors", []) == []
    res.notes["errors"] = "; ".join(out1.get("errors", []))[:300]

    d1, d2 = out1.get("decision"), out2.get("decision")
    res.checks["deterministic"] = bool(
        d1 and d2 and d1.action == d2.action and d1.quantity == d2.quantity
    )

    ens, th = out1.get("ensemble"), out1.get("thesis")
    res.checks["conviction_is_ensemble"] = bool(
        ens and th and abs(th.conviction - ens.score) < 1e-9
    )

    sigs = out1.get("verified_signals", [])
    res.checks["provenance"] = all(
        (
            e.source_kind in ("model", "feature")
            and e.feature_hash == out1["feature_hash"]
            and (e.source_kind != "model" or e.model_id)
        )
        or (e.source_kind in ("web", "filing") and e.source_url and e.quote)
        for s in sigs
        for e in s.evidence
    )

    prose = (
        " ".join(t.text for t in out1.get("debate", []))
        + " "
        + (th.narrative if th else "")
        + " "
        + (d1.report if d1 else "")
    )
    allowed = allowed_numbers(
        sigs,
        {"score": f"{ens.score:.2f}" if ens else "", "weights": str(ens.weights) if ens else ""},
    )
    prop, v = out1.get("proposal"), out1.get("verdict")
    extra = " ".join(
        str(x)
        for x in ([prop.entry, prop.stop_loss, prop.target, prop.quantity] if prop else [])
        + ([v.quantity] if v else [])
        + ([d1.quantity] if d1 else [])
    )
    extra += " " + (d1.report.split("\n\n")[0] if d1 else "")  # templated line is code-generated
    allowed |= {_canon(x) for x in _NUM.findall(extra)}
    allowed |= {_canon(x) for x in _NUM.findall(" ".join(th.key_risks) if th else "")}
    nums = {_canon(x) for x in _NUM.findall(re.sub(r"\[S\d+\]", "", prose))}
    bad = nums - allowed
    res.checks["no_uncited_numbers"] = not bad
    res.notes["uncited"] = ", ".join(sorted(bad))[:200]

    if d1 and d1.action in ("BUY", "SELL") and prop and v:
        p = risk_rules.Proposal(
            symbol=prop.symbol,
            sector=sector_of(prop.symbol),
            module=prop.module,
            side=prop.side,
            entry=prop.entry,
            stop=prop.stop_loss,
            target=prop.target,
            quantity=d1.quantity,
            atr=out1["atr"],
            avg_turnover_cr=out1.get("avg_turnover_cr", 0.0),
            circuit_today=out1.get("circuit_today", False),
            days_to_results=out1.get("days_to_results"),
            ensemble_score=ens.score,
            model_age_days=0,
        )
        again = risk_rules.evaluate(p, ctx.ledger.portfolio_state(), ctx.cfg.risk[prop.module])
        res.checks["rules_never_violated"] = (
            again.outcome != "reject" and again.quantity >= d1.quantity
        )
    else:
        res.checks["rules_never_violated"] = True
    return res


def fast_path_latency(n_ticks: int = 50_000, symbols: int = 50) -> float:
    """Replay synthetic ticks through BarBuilder + ExitGuard; return p99 in microseconds."""
    from ptaa.fast.bars import BarBuilder
    from ptaa.fast.exit_guard import ExitGuard
    from ptaa.models import Position

    builder, guard = BarBuilder(["1m", "5m", "1h", "1d"]), ExitGuard()
    syms = [f"S{i}" for i in range(symbols)]
    t0 = datetime(2026, 11, 17, 9, 15, tzinfo=IST)
    for s in syms[::2]:
        guard.add(
            Position(
                symbol=s,
                module="intraday",
                side="BUY",
                quantity=10,
                avg_price=100.0,
                stop_loss=1.0,
                target=1e9,
                opened_at=t0,
            )
        )
    lat = []
    for i in range(n_ticks):
        tk = Tick(
            symbol=syms[i % symbols],
            ts=t0 + timedelta(seconds=i * 0.2),
            ltp=100 + (i % 13) * 0.1,
            volume=i,
            source="eval",
        )
        a = time.perf_counter_ns()
        builder.push(tk)
        guard.evaluate(tk)
        lat.append(time.perf_counter_ns() - a)
    lat.sort()
    return lat[int(len(lat) * 0.99)] / 1e3


def run_all(ctx: GraphContext, snapshot_dir: Path) -> tuple[list[EvalResult], float]:
    results = [run_eval(ctx, p) for p in sorted(snapshot_dir.glob("*.json"))]
    return results, fast_path_latency()
