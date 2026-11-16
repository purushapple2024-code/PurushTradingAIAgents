"""LangGraph wiring: parallel analysts, Verifier, regime, ensemble, research manager, trader,
risk loop (max 2), portfolio manager, executor. SQLite checkpointer for replay."""

from __future__ import annotations

import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from ptaa.graph.context import GraphContext
from ptaa.graph.nodes.analysts import fundamental_analyst, sequence_analyst, technical_analyst
from ptaa.graph.nodes.data import data_collector
from ptaa.graph.nodes.decision import (
    after_risk,
    executor,
    portfolio_manager,
    research_manager,
    risk_manager,
    trader,
)
from ptaa.graph.nodes.gates import ensemble, regime, verifier
from ptaa.graph.nodes.news import news_analyst
from ptaa.graph.state import GraphState
from ptaa.models import Module


def build_graph(ctx: GraphContext, checkpoint_path: Path | None = None) -> Any:
    g: StateGraph = StateGraph(GraphState)
    bind = lambda fn: partial(fn, ctx=ctx)  # noqa: E731

    g.add_node("data_collector", bind(data_collector))
    g.add_node("technical_analyst", bind(technical_analyst))
    g.add_node("fundamental_analyst", bind(fundamental_analyst))
    g.add_node("news_analyst", bind(news_analyst))
    g.add_node("sequence_analyst", bind(sequence_analyst))
    g.add_node("verifier", bind(verifier))
    g.add_node("regime", bind(regime))
    g.add_node("ensemble", bind(ensemble))
    g.add_node("research_manager", bind(research_manager))
    g.add_node("trader", bind(trader))
    g.add_node("risk_manager", bind(risk_manager))
    g.add_node("portfolio_manager", bind(portfolio_manager))
    g.add_node("executor", bind(executor))

    g.add_edge(START, "data_collector")
    # fan-out to analysts in parallel; LangGraph merges `signals` via the operator.add reducer
    g.add_edge("data_collector", "technical_analyst")
    g.add_edge("data_collector", "fundamental_analyst")
    g.add_edge("data_collector", "news_analyst")
    g.add_edge("data_collector", "sequence_analyst")
    g.add_edge(
        ["technical_analyst", "fundamental_analyst", "news_analyst", "sequence_analyst"], "verifier"
    )
    g.add_edge("verifier", "regime")
    g.add_edge("regime", "ensemble")
    g.add_edge("ensemble", "research_manager")
    g.add_edge("research_manager", "trader")
    g.add_edge("trader", "risk_manager")
    g.add_conditional_edges(
        "risk_manager",
        after_risk,
        {"risk_manager": "risk_manager", "portfolio_manager": "portfolio_manager"},
    )
    g.add_edge("portfolio_manager", "executor")
    g.add_edge("executor", END)

    checkpointer = None
    if checkpoint_path is not None:
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpointer = SqliteSaver(
                sqlite3.connect(str(checkpoint_path), check_same_thread=False)
            )
        except ImportError:  # langgraph-checkpoint-sqlite not installed: run without replay
            checkpointer = None
    return g.compile(checkpointer=checkpointer)


def run_once(
    graph: Any,
    ctx: GraphContext,
    symbol: str,
    module: Module,
    snapshot: dict[str, Any],
    run_id: str | None = None,
) -> GraphState:
    run_id = run_id or uuid.uuid4().hex[:10]
    state: GraphState = {  # type: ignore[typeddict-item]
        "run_id": run_id,
        "symbol": symbol.upper(),
        "module": module,
        "signals": [],
        "debate": [],
        "risk_loops": 0,
        "audit": [],
        "errors": [],
        **snapshot,
    }
    if "as_of" not in state:
        state["as_of"] = datetime.now()
    out: GraphState = graph.invoke(state, config={"configurable": {"thread_id": run_id}})
    ctx.ledger.record_audit(run_id, out.get("audit", []))
    return out
