# PurushTradingAIAgents (`ptaa`)

Zero-running-cost, ML-first, multi-agent **paper trading** system for NSE equities.
Machine-learning models do the analysis; a small local open-weight LLM (Ollama) only explains,
debates and arbitrates; rules in Python decide. Two strategy modules (swing and intraday) share one
sub-millisecond fast path fed by free broker WebSockets (Angel One, Upstox) with a delayed
yfinance fallback.

> Paper trading by default. Nothing here is investment advice. Live trading exists but is gated five
> ways (see "Live execution"), has never been exercised against a real account in this repository, and
> is your responsibility to enable only after a long, honest paper record and a check of your broker's
> SEBI algo-trading terms.

Design doc and terminal UI design live in Claude artifacts (linked in the project conversation).

## Status: all 9 milestones built (paper by default; live gated)

| Milestone | Status |
| --- | --- |
| 1. Skeleton, data layer, fast path (feeds, bars, features, archive, exit guard), fees, rules, CLI | done |
| 2. Training pipeline v1 (triple-barrier labels, purged walk-forward, LightGBM, SHAP, registry, backtest, Colab notebook) | done |
| 3. Research graph (LangGraph): parallel ML analysts, Verifier, regime, ensemble, trader, risk loop, PM, paper executor, audit | done |
| 4. News and sentiment: ddgs + RSS discovery, allow-listed fetch, FinBERT with lexicon fallback, quote-verified evidence | done |
| 5. LLM layer: pluggable backends (Ollama / in-process / hosted-free / none), debate, narrative, personas, report, number guard; Colab demo | done |
| 6. Control plane (FastAPI, versioned settings/strategies/agents, kill switch), Streamlit terminal (5 screens), Telegram alerts | done |
| 7. Sequence model (LSTM → ONNX, Gaussian-NLL uncertainty) and the live engine with the intraday module | done |
| 8. PPO sizer (Gymnasium env, SB3 trainer, ONNX policy, used only if it beat fixed sizing), eval suite, CI gates, release scripts | done |
| 9. Gated live execution: env flag + live mode + adapter credentials + daily cap + per-order human approval; Angel One / Upstox / Flattrade adapters; TradingView webhook input | done (adapters untested against live accounts) |

## Quick start

```bash
cd PurushTradingAIAgents
uv venv && source .venv/bin/activate        # or: python -m venv .venv
uv pip install -e ".[dev]"                   # or: pip install -e ".[dev]"
cp .env.example .env                         # optional; runs on yfinance without any keys

ptaa --help
ptaa settings                                # effective settings, secrets masked
ptaa universe NIFTY50
ptaa data RELIANCE --interval 1d --tail 5    # cached yfinance candles
ptaa features RELIANCE                       # deterministic feature row + hashes
ptaa fees 1000 100 --module swing            # Indian cash-equity fee breakdown
ptaa live --simulate --seconds 10            # fast path on a seeded simulated feed
ptaa live --module intraday                  # real feed chain (broker WS if keys, else yfinance polling)

# Training (milestone 2)
ptaa train gbm --horizon 5d                  # walk-forward LightGBM on NIFTY 50 daily bars (yfinance)
ptaa train gbm --synthetic                   # same pipeline on seeded synthetic data, offline, ~2 s
ptaa models list && ptaa models verify gbm-5d-v1
ptaa models explain gbm-5d-v1 RELIANCE       # P(down/flat/up) + SHAP top features for the latest bar

# Research graph (milestone 3)
ptaa run RELIANCE --module swing             # analysts → verifier → regime → ensemble → trader → risk → PM → paper fill
ptaa run RELIANCE --synthetic                # same on synthetic data, offline
ptaa portfolio                               # positions, cash, equity, fees, recent runs
ptaa replay <run_id>                         # per-node audit trail with latencies and hashes
ptaa run RELIANCE --no-news                  # skip web discovery (offline)
ptaa run RELIANCE --no-llm                   # templates instead of the local LLM; identical decisions

# Terminal + control plane (milestone 6)
ptaa serve                                   # FastAPI at 127.0.0.1:8765/docs, Streamlit terminal at 127.0.0.1:8501

# Live engine (milestone 7)
ptaa live --module intraday --trade          # feed → bars → exit guard; research graph on each closed 5m bar (top-N); paper fills
ptaa live --simulate --sim-step 2 --trade --seconds 60   # replay a whole simulated session in a minute
ptaa train seq --horizon 5                   # LSTM sequence analyst (PyTorch; Colab/Kaggle), exported to ONNX

# Evals and releases (milestone 8)
ptaa eval --record 30                        # frozen snapshots: no uncited numbers, conviction == ensemble, rules re-verified, deterministic, p99 latency
scripts/release_models.sh v0.1.0             # upload model files as GitHub release assets (needs gh)
scripts/fetch_models.sh v0.1.0               # download + verify on the laptop
```

PPO sizer: `training/notebooks/train_ppo_colab.ipynb` trains a Stable-Baselines3 policy in a Gymnasium
paper environment where the agent only picks a risk fraction; the trader uses it only when its
out-of-sample run beat fixed-fraction sizing (`beats_fixed == 1`), and the rule engine still caps every size.

The engine re-reads engine state and risk limits from the control plane on every decision bar, so
pausing a module, changing a limit or hitting the kill switch in the terminal takes effect immediately.
Intraday entries are allowed only inside the configured window and all intraday positions are squared
off at 15:10 by the exit guard.

The terminal lets you add/edit/delete strategies and agents (scope, model binding, permissions), edit risk
limits per module, switch the LLM backend, pause modules and hit the kill switch. Every change is a new
version in `control.db` with an audit row and one-click revert. Live execution mode is locked in v1.
Telegram alerts are on by default and need `PTAA_TELEGRAM_BOT_TOKEN` and `PTAA_TELEGRAM_CHAT_ID`.

LLM: `ollama pull qwen3:4b` (about 2.5 GB, fits a 4 GB budget; `llama3.2:3b` as fallback) and set
`PTAA_LLM_BACKEND=ollama`. The LLM only writes the debate, thesis narrative, risk persona views and
the report. Sentences containing numbers that are not in the verified signals, or citing unknown
signal ids, are stripped before they enter state; conviction is always the ensemble score. With
`PTAA_LLM_BACKEND=none` every decision is identical, only the prose disappears.
`training/notebooks/demo_full_graph_colab.ipynb` runs the whole graph with an in-process llama.cpp model on a free GPU.

News uses DuckDuckGo news search and outlet RSS feeds (no API keys), fetches only allow-listed
domains, and FinBERT (`pip install -e ".[nlp]"`) when available, else a transparent lexicon.
Every news claim carries its URL and a verbatim lead quote; the Verifier re-reads the cached page and
drops any claim whose quote is not in the text.

Colab/Kaggle: open `training/notebooks/train_gbm_colab.ipynb` (badge inside) to train on free GPUs and keep models in Drive.

Broker feeds need the `brokers` extra and credentials in `.env`:
`uv pip install -e ".[dev,brokers]"`. Upstox v3 additionally needs the generated protobuf module
(`ptaa auth upstox --fetch-proto`, milestone 2).

## Layout

```
src/ptaa/
  settings.py        # pydantic-settings (PTAA_* env) + validated config.yaml models
  models.py          # Tick, Bar, Position, ExitSignal
  data/              # MarketFeed/BrokerAdapter protocols, yfinance, NSE public, cache, universes
  fast/              # bars, features, archive, exit_guard, feeds/ (angelone_ws, upstox_ws, failover, simulated)
  risk/              # fees, rules (15 hard rules), sizing
  mlmodels/          # versioned model registry with hash + feature-schema verification; seq.py (ONNX inference)
  training/          # labels (triple-barrier), splits (purged walk-forward), dataset, train_gbm, backtest, pipeline
  graph/             # LangGraph state, context + audit wrapper, nodes/ (analysts, gates, decision, data), graph.py
  ledger/            # SQLite paper ledger: runs, audit, orders, positions, equity; PaperBroker
  execution/         # ExecutionBroker protocol, PaperBroker (ledger), LiveGate, live_brokers (Angel One, Upstox, Flattrade), TradingView webhook input
  research/          # guarded HTTP (allow-list, cache), discover (ddgs, RSS), extract, sentiment (FinBERT/lexicon)
  llm/               # LLMClient (ollama | inprocess | hosted_free | none), versioned prompts/, reasoning guards
  control/           # ControlService (SQLite docs with version history + audit), FastAPI app
  evals.py           # frozen-snapshot eval suite: provenance, determinism, number guard, rules, latency
  engine.py          # LiveEngine: feed → bars → exit guard; research graph on closed decision bars; control-plane aware
  alerts/            # Telegram alerts (fills, rejects, halts, feed, summary)
  cli.py
training/notebooks/  # Colab/Kaggle notebooks
terminal/            # Streamlit terminal (Live, Strategies, Agents, Decisions, Control), dark theme
tests/               # 59 tests incl. no-lookahead, purge/embargo, tamper detection, end-to-end training
config.yaml          # seed configuration; runtime settings move to the control plane in milestone 6
```

## Principles

- No number that reaches the ledger originates in an LLM.
- Every signal carries provenance (model id, feature-row hash, source URL); the Verifier drops the rest.
- Honest backtests: walk-forward, purged folds, fees and slippage, baseline next to every result.
- Secrets only via environment; masked everywhere; analysts have no network and no ledger access.

## Development

```bash
ruff check src tests && ruff format src tests
pytest -q
mypy src
```

## Licence

MIT. Third-party repos consulted (TradingAgents, FinRL, Qlib, FinBERT, ML4T) are credited in the
design doc; code is adapted, not vendored, and each licence is checked before reuse.

## Live execution (milestone 9) — read before enabling

A live order is placed only when **all** of the following hold, checked in this order by `LiveGate`:

1. `PTAA_LIVE_TRADING_ENABLED=true` is set in the environment. It cannot be set from the UI, the API, the LLM or a config file.
2. The control plane's engine state has `execution_mode == "live"` and the kill switch is off. Switching to live is refused unless step 1 is already true.
3. The selected `execution_adapter` (angelone, upstox, flattrade) has credentials in the environment.
4. The order would not push today's live notional past `settings/live.daily_notional_cap_inr` (default ₹50,000).
5. A human approved **this exact order** (symbol, side, quantity) within 10 minutes, in the terminal (Control → Live execution) or via the API (`POST /approvals/{id}/approve`). Telegram announces every pending approval.

Everything else stays identical to paper mode: same models, same rules, same ledger and audit. The broker adapters follow each API's documented order endpoints but have not been run against live accounts here; make your first live order quantity 1.

TradingView has no order API, so `POST /webhooks/tradingview` (shared secret) records alerts as low-weight external hints and notifications. It never places orders.

## Running on your laptop vs. Colab

* **Laptop** (8 GB RAM is enough): the live engine, terminal, control plane, Ollama (Qwen3 4B ≈ 2.5 GB) and ONNX inference. Yahoo/broker access needed for real data.
* **Colab/Kaggle** (free GPU): the three training notebooks (GBM, LSTM, PPO) and the full-graph demo with an in-process llama.cpp model. Sessions time out, so they are not for live trading.
