"""PurushTradingAIAgents terminal (Streamlit). Dark obsidian, high density, five screens.

Reads the paper ledger and the control plane directly (same SQLite files) so it works without the
HTTP API running; every change goes through ControlService and is versioned and audited.
Run: `ptaa serve` or `streamlit run terminal/app.py`.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from ptaa.control.service import AgentDoc, ControlService, StrategyDoc
from ptaa.ledger.db import Ledger
from ptaa.mlmodels.registry import ModelRegistry
from ptaa.settings import Settings, load_config

st.set_page_config(
    page_title="PurushTradingAIAgents Terminal", layout="wide", initial_sidebar_state="collapsed"
)

CSS = """
<style>
:root { --bg:#0A0C0F; --panel:#0E1115; --line:#1C2128; --ink:#E6E9ED; --mute:#9AA3AE; --dim:#5E6873;
        --up:#3DD68C; --dn:#F0616D; --amb:#F5B544; --acc:#5B9CF6; }
html, body, [class*="css"] { font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; font-size: 12px; }
.block-container { padding: 0.6rem 1rem 1rem 1rem; max-width: 100%; }
h1, h2, h3 { font-weight: 600 !important; letter-spacing: 0.02em; }
h2 { font-size: 13px !important; color: var(--mute) !important; text-transform: uppercase; letter-spacing: .08em; margin: .4rem 0 .3rem 0 !important; }
.kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; }
.kpi .l { color: var(--dim); font-size: 10px; letter-spacing: .08em; }
.kpi .v { color: var(--ink); font-size: 18px; margin-top: 2px; }
.kpi .s { font-size: 11px; color: var(--mute); }
.up { color: var(--up) !important; } .dn { color: var(--dn) !important; } .amb { color: var(--amb) !important; } .acc { color: var(--acc) !important; }
.tag { display:inline-block; padding: 2px 8px; border-radius: 3px; border: 1px solid #27303B; background:#1A2029; color: var(--amb); font-size: 11px; }
.kill button { background:#2A1416 !important; border:1px solid #7A2A31 !important; color:#F0848D !important; font-weight:600 !important; }
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
footer, #MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def _boot():
    s = Settings()  # fresh read of the environment each boot
    cfg = load_config(os.environ.get("PTAA_CONFIG", "config.yaml"))
    data = Path(s.data_dir)
    svc = ControlService(data / "control.db", cfg)
    ledger = Ledger(data / "ledger.db", cfg.fees, cfg.paper_book_inr)
    reg = ModelRegistry(data / "models")
    return s, cfg, svc, ledger, reg


S, CFG, SVC, LEDGER, REG = _boot()
ENGINE = SVC.engine_state()
ACTOR = os.environ.get("USER", "operator")


def kpi(label: str, value: str, sub: str = "", cls: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="l">{label}</div><div class="v {cls}">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- header
hl, hm, hr = st.columns([3, 6, 2])
with hl:
    st.markdown(
        "**PurushTradingAIAgents** <span style='color:#5E6873'>Terminal</span>",
        unsafe_allow_html=True,
    )
with hm:
    mode = ENGINE.execution_mode.upper()
    st.markdown(
        f"<span style='color:#9AA3AE'>IST {datetime.now().strftime('%H:%M:%S')} · LLM <span style='color:#E6E9ED'>{S.llm_model}</span> {S.llm_backend} · "
        f"feeds angel={'on' if S.has_angel else 'off'} upstox={'on' if S.upstox_access_token else 'off'} · "
        f"swing {'running' if ENGINE.swing_running else 'stopped'} · intraday {'running' if ENGINE.intraday_running else 'stopped'}</span> "
        f"<span class='tag'>{mode}</span>"
        + (
            " <span class='tag' style='color:#F0848D'>KILL SWITCH ON</span>"
            if ENGINE.kill_switch
            else ""
        ),
        unsafe_allow_html=True,
    )
with hr:
    st.markdown('<div class="kill">', unsafe_allow_html=True)
    if st.button("KILL SWITCH", use_container_width=True):
        SVC.kill(actor=ACTOR)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

tab_live, tab_strat, tab_agents, tab_dec, tab_ctrl = st.tabs(
    ["Live", "Strategies", "Agents", "Decisions", "Control"]
)

# ----------------------------------------------------------------------------- Live
with tab_live:
    cash, value, eq = LEDGER.equity()
    peak = LEDGER.peak_equity()
    dd = (eq / peak - 1) * 100 if peak else 0.0
    gross = value / eq * 100 if eq else 0.0
    runs = LEDGER.runs(50)
    cols = st.columns(8)
    with cols[0]:
        kpi(
            "EQUITY",
            f"₹{eq:,.0f}",
            f"{(eq / CFG.paper_book_inr - 1) * 100:+.2f}% since start",
            "up" if eq >= CFG.paper_book_inr else "dn",
        )
    with cols[1]:
        kpi("DAY P&L", f"₹{LEDGER.day_pnl():+,.0f}", "", "up" if LEDGER.day_pnl() >= 0 else "dn")
    with cols[2]:
        kpi("CASH", f"₹{cash:,.0f}", f"positions ₹{value:,.0f}")
    with cols[3]:
        kpi(
            "DRAWDOWN",
            f"{dd:.2f}%",
            f"halt at -{CFG.risk['swing'].drawdown_halt_pct:.0f}%",
            "dn" if dd < -5 else "",
        )
    with cols[4]:
        kpi(
            "GROSS EXPOSURE",
            f"{gross:.0f}%",
            f"cap {CFG.risk['swing'].max_gross_exposure_pct:.0f}%",
        )
    with cols[5]:
        last_reg = next((r.regime for r in runs if r.regime), "—")
        kpi(
            "REGIME",
            last_reg.replace("_", " ").upper() if last_reg else "—",
            "from last run",
            "acc",
        )
    with cols[6]:
        kpi("FEES PAID", f"₹{LEDGER.fees_paid():,.0f}", "all time")
    with cols[7]:
        models = REG.list()
        fresh = sum(1 for m in models if REG.verify(m.model_id)[0])
        kpi("MODELS", f"{fresh} / {len(models)}", "verified", "amb" if fresh < len(models) else "")

    left, right = st.columns([7, 5])
    with left:
        st.markdown("## Open positions")
        pos = LEDGER.positions()
        if pos:
            df = pd.DataFrame([p.model_dump() for p in pos])
            df["pnl"] = (
                (df["last_price"].fillna(df["avg_price"]) - df["avg_price"]) * df["quantity"]
            ).round(0)
            st.dataframe(
                df[
                    [
                        "symbol",
                        "module",
                        "side",
                        "quantity",
                        "avg_price",
                        "last_price",
                        "pnl",
                        "stop_loss",
                        "target",
                        "strategy",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                height=220,
            )
        else:
            st.caption("no open positions")
        st.markdown("## Recent decisions")
        if runs:
            rdf = pd.DataFrame([r.model_dump() for r in runs])[
                [
                    "run_id",
                    "symbol",
                    "module",
                    "decision",
                    "quantity",
                    "ensemble_score",
                    "regime",
                    "verdict",
                    "created_at",
                ]
            ]
            st.dataframe(rdf, use_container_width=True, hide_index=True, height=300)
        else:
            st.caption("no runs yet — try `ptaa run RELIANCE`")
    with right:
        st.markdown("## Agent health")
        agents = SVC.list("agents")
        rows = [
            {
                "agent": k,
                "layer": v["layer"],
                "kind": v["kind"],
                "enabled": "on" if v["enabled"] else "off",
                "model": v.get("model_id") or "",
            }
            for k, v in agents.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=560)

# ----------------------------------------------------------------------------- Strategies
with tab_strat:
    strategies = SVC.list("strategies")
    cl, cr = st.columns([6, 5])
    with cl:
        st.markdown(
            f"## Strategies · {len(strategies)} defined · {sum(1 for s in strategies.values() if s['enabled'])} enabled"
        )
        if strategies:
            sdf = pd.DataFrame(
                [
                    {
                        "name": k,
                        **{
                            f: v.get(f)
                            for f in (
                                "module",
                                "universe",
                                "status",
                                "enabled",
                                "decision_bar",
                                "ensemble_floor",
                                "stop_atr",
                                "target_atr",
                                "sizer",
                            )
                        },
                        "signals": ", ".join(v.get("signals", [])),
                        "v": SVC.version_of("strategies", k),
                    }
                    for k, v in strategies.items()
                ]
            )
            st.dataframe(sdf, use_container_width=True, hide_index=True)
        with st.expander("+ New strategy"):
            with st.form("new_strategy"):
                name = st.text_input("Name")
                module = st.selectbox("Module", ["swing", "intraday"])
                universe = st.selectbox(
                    "Universe", ["NIFTY50", "NIFTY100", "NIFTY50_LIQUID15", "custom"]
                )
                if st.form_submit_button("Create draft") and name:
                    SVC.put(
                        "strategies",
                        name,
                        StrategyDoc(name=name, module=module, universe=universe).model_dump(),
                        actor=ACTOR,
                        note="created in terminal",
                    )
                    st.success(f"created {name} v1")
                    st.rerun()
    with cr:
        if strategies:
            sel = st.selectbox("Edit strategy", list(strategies), key="strat_sel")
            doc = StrategyDoc.model_validate(strategies[sel])
            st.caption(f"v{SVC.version_of('strategies', sel)} · {doc.status}")
            with st.form("edit_strategy"):
                c1, c2 = st.columns(2)
                enabled = c1.checkbox("Enabled", doc.enabled)
                status = c2.selectbox(
                    "Status",
                    ["draft", "backtest", "shadow", "live_paper"],
                    index=["draft", "backtest", "shadow", "live_paper"].index(doc.status),
                )
                universe = c1.selectbox(
                    "Universe",
                    ["NIFTY50", "NIFTY100", "NIFTY50_LIQUID15", "custom"],
                    index=["NIFTY50", "NIFTY100", "NIFTY50_LIQUID15", "custom"].index(doc.universe)
                    if doc.universe in ("NIFTY50", "NIFTY100", "NIFTY50_LIQUID15", "custom")
                    else 0,
                )
                bar = c2.selectbox(
                    "Decision bar",
                    ["1d", "1h", "5m"],
                    index=["1d", "1h", "5m"].index(doc.decision_bar)
                    if doc.decision_bar in ("1d", "1h", "5m")
                    else 0,
                )
                signals = st.multiselect(
                    "Signal sources", ["technical", "sequence", "news", "fundamental"], doc.signals
                )
                c3, c4, c5 = st.columns(3)
                floor = c3.number_input("Ensemble ≥", 0.0, 1.0, float(doc.ensemble_floor), 0.01)
                stop_atr = c4.number_input("Stop × ATR", 0.1, 10.0, float(doc.stop_atr), 0.1)
                target_atr = c5.number_input("Target × ATR", 0.1, 20.0, float(doc.target_atr), 0.1)
                hold = c3.number_input("Max hold (days)", 1, 60, int(doc.max_hold_days))
                trail = c4.number_input(
                    "Trail after (R)", 0.0, 5.0, float(doc.trail_after_r or 0.0), 0.1
                )
                sizer = c5.selectbox(
                    "Sizer", ["fixed_risk", "ppo"], index=["fixed_risk", "ppo"].index(doc.sizer)
                )
                gate = st.multiselect(
                    "Regime gate", ["trend_up", "trend_down", "range", "high_vol"], doc.regime_gate
                )
                extra = st.text_area("Advanced (YAML)", doc.yaml_extra, height=90)
                b1, b2, b3 = st.columns(3)
                if b1.form_submit_button("Save new version"):
                    new = doc.model_copy(
                        update=dict(
                            enabled=enabled,
                            status=status,
                            universe=universe,
                            decision_bar=bar,
                            signals=signals,
                            ensemble_floor=floor,
                            stop_atr=stop_atr,
                            target_atr=target_atr,
                            max_hold_days=hold,
                            trail_after_r=trail or None,
                            sizer=sizer,
                            regime_gate=gate,
                            yaml_extra=extra,
                        )
                    )
                    try:
                        v = SVC.put(
                            "strategies",
                            sel,
                            new.model_dump(),
                            actor=ACTOR,
                            note="edited in terminal",
                        )
                        st.success(f"saved v{v}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                if b2.form_submit_button("Delete"):
                    SVC.delete("strategies", sel, actor=ACTOR)
                    st.rerun()
                if b3.form_submit_button("Revert one version"):
                    v = SVC.version_of("strategies", sel)
                    if v > 1:
                        SVC.revert("strategies", sel, v - 1, actor=ACTOR)
                        st.rerun()
            with st.expander("History"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "v": h.version,
                                "actor": h.actor,
                                "note": h.note,
                                "deleted": h.deleted,
                                "at": h.at,
                            }
                            for h in SVC.history("strategies", sel)
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

# ----------------------------------------------------------------------------- Agents
with tab_agents:
    agents = SVC.list("agents")
    al, ar = st.columns([6, 5])
    with al:
        st.markdown(f"## Agents · {len(agents)} nodes")
        for layer in ("fast", "analysts", "gates", "reasoning", "decision", "execution"):
            rows = [
                {
                    "agent": k,
                    "kind": v["kind"],
                    "enabled": "on" if v["enabled"] else "off",
                    "runs on": v["runs_on"],
                    "modules": ",".join(v["scope_modules"]),
                    "model": v.get("model_id") or "",
                    "min conf": v.get("min_confidence"),
                    "max age d": v.get("max_model_age_days"),
                }
                for k, v in agents.items()
                if v["layer"] == layer
            ]
            if rows:
                st.markdown(f"## {layer}")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with st.expander("+ New agent"):
            with st.form("new_agent"):
                name = st.text_input("Name")
                kind = st.selectbox("Kind", ["ml", "llm", "code"])
                layer = st.selectbox(
                    "Layer", ["fast", "analysts", "gates", "reasoning", "decision", "execution"]
                )
                if st.form_submit_button("Create") and name:
                    SVC.put(
                        "agents",
                        name,
                        AgentDoc(name=name, kind=kind, layer=layer, enabled=False).model_dump(),
                        actor=ACTOR,
                    )
                    st.rerun()
    with ar:
        if agents:
            sel = st.selectbox("Edit agent", list(agents), key="agent_sel")
            doc = AgentDoc.model_validate(agents[sel])
            models = [m.model_id for m in REG.list()]
            with st.form("edit_agent"):
                c1, c2 = st.columns(2)
                enabled = c1.checkbox("Enabled", doc.enabled)
                runs_on = c2.selectbox(
                    "Runs on",
                    ["every_closed_bar", "daily_after_close", "on_demand"],
                    index=["every_closed_bar", "daily_after_close", "on_demand"].index(doc.runs_on),
                )
                mods = st.multiselect("Scope · modules", ["swing", "intraday"], doc.scope_modules)
                unis = st.multiselect(
                    "Scope · universes",
                    ["NIFTY50", "NIFTY100", "NIFTY50_LIQUID15"],
                    [
                        u
                        for u in doc.scope_universe
                        if u in ("NIFTY50", "NIFTY100", "NIFTY50_LIQUID15")
                    ],
                )
                hors = st.multiselect("Scope · horizons", ["1h", "1d", "5d"], doc.scope_horizons)
                model_id = st.selectbox(
                    "Registered model",
                    ["(none)"] + models,
                    index=(["(none)"] + models).index(doc.model_id)
                    if doc.model_id in models
                    else 0,
                )
                c3, c4 = st.columns(2)
                max_age = c3.number_input(
                    "Max model age (days)", 0, 365, int(doc.max_model_age_days or 0)
                )
                min_conf = c4.number_input(
                    "Min confidence", 0.0, 1.0, float(doc.min_confidence or 0.0), 0.01
                )
                st.caption(
                    "Permissions (analysts never get network beyond allow-listed research; only the Executor writes the ledger)"
                )
                p1, p2, p3, p4 = st.columns(4)
                perms = {
                    "read_features": p1.checkbox(
                        "read features", doc.permissions.get("read_features", True)
                    ),
                    "emit_signals": p2.checkbox(
                        "emit signals", doc.permissions.get("emit_signals", True)
                    ),
                    "network": p3.checkbox("network", doc.permissions.get("network", False)),
                    "ledger_write": p4.checkbox(
                        "ledger write",
                        doc.permissions.get("ledger_write", False),
                        disabled=doc.layer != "execution",
                    ),
                }
                b1, b2 = st.columns(2)
                if b1.form_submit_button("Save"):
                    new = doc.model_copy(
                        update=dict(
                            enabled=enabled,
                            runs_on=runs_on,
                            scope_modules=mods,
                            scope_universe=unis,
                            scope_horizons=hors,
                            model_id=None if model_id == "(none)" else model_id,
                            max_model_age_days=max_age or None,
                            min_confidence=min_conf or None,
                            permissions=perms,
                        )
                    )
                    v = SVC.put(
                        "agents", sel, new.model_dump(), actor=ACTOR, note="edited in terminal"
                    )
                    st.success(f"saved v{v}")
                    st.rerun()
                if b2.form_submit_button("Delete"):
                    SVC.delete("agents", sel, actor=ACTOR)
                    st.rerun()
            st.markdown("## Model registry")
            if models:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "model": m.model_id,
                                "train_end": m.train_end,
                                "rows": m.n_samples,
                                "bal.acc": round(
                                    m.metrics.get("balanced_accuracy", float("nan")), 3
                                ),
                                "top-decile": round(
                                    m.metrics.get("top_decile_precision_up", float("nan")), 3
                                ),
                                "age d": REG.freshness_days(m),
                                "ok": REG.verify(m.model_id)[0],
                            }
                            for m in REG.list()
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("no models registered — run `ptaa train gbm`")

# ----------------------------------------------------------------------------- Decisions
with tab_dec:
    runs = LEDGER.runs(100)
    if not runs:
        st.caption("no runs yet")
    else:
        sel = st.selectbox(
            "Run",
            [
                f"{r.run_id} · {r.symbol} · {r.module} · {r.decision} · {r.created_at:%d %b %H:%M}"
                for r in runs
            ],
        )
        run = runs[
            [
                f"{r.run_id} · {r.symbol} · {r.module} · {r.decision} · {r.created_at:%d %b %H:%M}"
                for r in runs
            ].index(sel)
        ]
        audit = LEDGER.audit_for(run.run_id)
        steps = st.columns(max(1, len(audit)))
        for i, a in enumerate(audit):
            with steps[i]:
                kpi(
                    f"{i + 1:02d} {a.node.replace('_', ' ')[:16]}",
                    f"{a.duration_ms:.0f} ms",
                    a.note or "ok",
                    "dn" if a.note else "",
                )
        d1, d2 = st.columns([7, 5])
        with d1:
            st.markdown("## Decision report")
            st.markdown(
                f"**{run.decision} {run.quantity or 0} {run.symbol}** · ensemble {run.ensemble_score or 0:.2f} · regime {run.regime} · verdict {run.verdict or '—'}"
            )
            st.text(run.report or "")
        with d2:
            st.markdown("## Audit trail")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "node": a.node,
                            "ms": round(a.duration_ms, 1),
                            "in": a.inputs_hash,
                            "out": a.outputs_hash,
                            "model": a.model_id or "",
                            "note": a.note,
                        }
                        for a in audit
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

# ----------------------------------------------------------------------------- Control
with tab_ctrl:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("## Engine")
        e = SVC.engine_state()
        with st.form("engine"):
            sw = st.checkbox("Swing running", e.swing_running)
            idt = st.checkbox("Intraday running", e.intraday_running)
            kill = st.checkbox("Kill switch", e.kill_switch)
            mode = st.selectbox(
                "Execution mode",
                ["paper", "shadow", "live (locked)"],
                index=["paper", "shadow"].index(e.execution_mode)
                if e.execution_mode in ("paper", "shadow")
                else 0,
            )
            adapter = st.selectbox(
                "Execution adapter",
                ["paper", "angelone", "upstox", "flattrade"],
                index=["paper", "angelone", "upstox", "flattrade"].index(e.execution_adapter),
            )
            if st.form_submit_button("Apply"):
                try:
                    SVC.set_engine(
                        actor=ACTOR,
                        swing_running=sw and not kill,
                        intraday_running=idt and not kill,
                        kill_switch=kill,
                        execution_mode=mode.split(" ")[0],
                        execution_adapter=adapter,
                    )
                    st.success("applied")
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))
        st.markdown("## Feeds")
        st.json(SVC.get("settings", "feeds") or {}, expanded=False)
        st.markdown("## Universe")
        st.json(SVC.get("settings", "universe") or {}, expanded=False)
    with c2:
        st.markdown("## Reasoning LLM")
        llm = SVC.get("settings", "llm") or {}
        with st.form("llm"):
            backend = st.selectbox(
                "Backend",
                ["ollama", "inprocess", "hosted_free", "none"],
                index=["ollama", "inprocess", "hosted_free", "none"].index(
                    llm.get("backend", "ollama")
                ),
            )
            model = st.text_input("Model", llm.get("model", "qwen3:4b"))
            fb = st.text_input("Fallback model", llm.get("fallback_model", "llama3.2:3b"))
            rounds = st.number_input("Debate rounds", 0, 4, int(llm.get("debate_rounds", 2)))
            if st.form_submit_button("Save LLM settings"):
                v = SVC.put(
                    "settings",
                    "llm",
                    {
                        **llm,
                        "backend": backend,
                        "model": model,
                        "fallback_model": fb,
                        "debate_rounds": rounds,
                    },
                    actor=ACTOR,
                )
                st.success(f"saved v{v}")
        st.markdown("## Schedule")
        st.json(SVC.get("settings", "schedule") or {}, expanded=False)
        st.markdown("## Alerts")
        alerts = SVC.get("settings", "alerts") or {}
        with st.form("alerts"):
            tg = st.checkbox("Telegram", alerts.get("telegram", True))
            dk = st.checkbox("Desktop", alerts.get("desktop", True))
            em = st.checkbox("Daily report email", alerts.get("daily_report_email", False))
            if st.form_submit_button("Save alerts"):
                SVC.put(
                    "settings",
                    "alerts",
                    {"telegram": tg, "desktop": dk, "daily_report_email": em},
                    actor=ACTOR,
                )
                st.success("saved")
        st.caption(
            f"Telegram configured: {'yes' if S.telegram_bot_token and S.telegram_chat_id else 'no (set PTAA_TELEGRAM_*)'}"
        )
    with c3:
        st.markdown("## Risk limits · enforced in code")
        for module in ("swing", "intraday"):
            lim = SVC.get("settings", f"risk_{module}") or {}
            with st.form(f"risk_{module}"):
                st.caption(f"{module} · v{SVC.version_of('settings', f'risk_{module}')}")
                new = {}
                cols = st.columns(2)
                for i, (k, v) in enumerate(lim.items()):
                    if isinstance(v, bool) or v is None:
                        new[k] = v
                        continue
                    new[k] = cols[i % 2].number_input(
                        k, value=float(v), key=f"{module}_{k}", format="%.3f"
                    )
                    if isinstance(v, int):
                        new[k] = int(new[k])
                b1, b2 = st.columns(2)
                if b1.form_submit_button(f"Apply {module}"):
                    try:
                        ver = SVC.put(
                            "settings",
                            f"risk_{module}",
                            new,
                            actor=ACTOR,
                            note="risk limits from terminal",
                        )
                        st.success(f"applied v{ver}")
                    except ValueError as err:
                        st.error(str(err))
                if b2.form_submit_button(f"Revert {module}"):
                    ver = SVC.version_of("settings", f"risk_{module}")
                    if ver > 1:
                        SVC.revert("settings", f"risk_{module}", ver - 1, actor=ACTOR)
                        st.rerun()
        st.markdown("## Secrets · masked")
        st.dataframe(
            pd.DataFrame(
                [
                    {"key": k, "value": v}
                    for k, v in S.safe_dict().items()
                    if k.startswith(("angel", "upstox", "flattrade", "telegram", "hosted"))
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("## Live execution · gate and approvals")
    from ptaa.execution.gate import LiveGate

    gate = LiveGate(S, SVC)
    ok, why = gate.environment_ok()
    st.markdown(
        f"live possible: **{'yes' if ok else 'no'}** · {why} · daily cap ₹{gate.daily_cap_inr():,.0f} · "
        f"live notional today ₹{gate.notional_today():,.0f}"
    )
    pend = gate.pending()
    if not pend:
        st.caption("no pending approvals")
    for ap in pend:
        o = ap.order
        c1, c2, c3 = st.columns([6, 1, 1])
        c1.markdown(
            f"`{ap.approval_id}` {o['side']} {o['quantity']} {o['symbol']} ({o['module']}) · "
            f"notional ₹{ap.notional:,.0f} · stop {o['stop_loss']} target {o['target']} · {ap.created_at:%H:%M:%S}"
        )
        if c2.button("Approve", key=f"ap_{ap.approval_id}"):
            gate.decide(ap.approval_id, True, by=ACTOR)
            st.rerun()
        if c3.button("Reject", key=f"rj_{ap.approval_id}"):
            gate.decide(ap.approval_id, False, by=ACTOR)
            st.rerun()

    st.markdown("## Control audit")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "at": a.at,
                    "actor": a.actor,
                    "action": a.action,
                    "collection": a.collection,
                    "key": a.key,
                    "v": a.version,
                    "diff": a.diff[:120],
                }
                for a in SVC.audit(50)
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    f"paper only · not investment advice · session {uuid.uuid4().hex[:6]} · {datetime.now(tz=timezone.utc):%Y-%m-%d %H:%M UTC}"
)
