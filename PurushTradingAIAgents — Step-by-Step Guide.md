# PurushTradingAIAgents — Step-by-Step Guide

Nov 16, 2026 · @Purushotham Ranganath

Follow these steps in order on your laptop. Each one ends with a check so you know it worked before moving on. Everything is paper trading until step 9, and step 9 is deliberately far away.

## Step 1 · Push the latest commit and check CI (10 minutes)

Your GitHub repo has the 9 milestone commits. One more commit (`b1b478a`, notebook links and badges) is only on your laptop in the project folder, because the sandbox cannot reach GitHub.

1. Open a terminal (Terminal on Mac, PowerShell on Windows) and go to the project folder, the same one you pushed from:

   ```bash
   cd PurushTradingAIAgents
   git status
   ```

   You should see `On branch main` and no red files.
2. Push the pending commit:

   ```bash
   git pull --rebase origin main
   git push origin main
   ```

   If `git pull` complains about identity, run `git config user.name "Purushotham"` and `git config user.email "<your github email>"` once, then repeat.
3. In the browser open https://github.com/purushapple2024-code/PurushTradingAIAgents/actions. A workflow named **ci** starts automatically on every push. Click the newest run.
4. Wait 5–10 minutes. Two jobs must be green: **test** (lint + 59 tests on Python 3.10/3.11/3.12) and **evals** (trains a synthetic model, runs the eval suite, replays a simulated trading session).

**If the push asks for a password and fails** ("Password authentication is not supported"): GitHub needs a token, not your account password. Easiest on a Mac: `brew install gh`, then `gh auth login` (choose GitHub.com → HTTPS → Yes → Login with a web browser, paste the one-time code), then `git push origin main` again. Alternative: create a classic Personal Access Token with the `repo` scope under GitHub Settings → Developer settings, run `git config --global credential.helper osxkeychain`, and paste the token when `git push` asks for the password.

**Check:** both jobs show a green tick. If a job is red, click it, copy the last 30 lines of the log, and paste them to me; I will fix the code.

**Also check once:** at https://github.com/purushapple2024-code/PurushTradingAIAgents there must be **no** file named `.env` and no `data/` folder. Those hold secrets and local data and are excluded by `.gitignore`; if you ever see them in the repo, tell me before doing anything else.

## Step 2 · Install on your laptop and run the offline checks (20 minutes)

Nothing here needs an account, a key or market hours. It proves the whole pipeline on your machine with synthetic data.

1. Install Python 3.11 if you do not have it (https://www.python.org/downloads/). Check: `python3 --version` prints 3.10 or higher.
2. Install `uv` (fast package manager): Mac/Linux `curl -LsSf https://astral.sh/uv/install.sh | sh`; Windows PowerShell `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`. Close and reopen the terminal.
3. In the project folder:

   ```bash
   uv venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]" onnx gymnasium
   cp .env.example .env                # Windows: copy .env.example .env
   ```

   Leave `.env` as it is for now; empty keys mean "paper mode, free data, local LLM off until step 5".
4. Run the test suite: `pytest -q`. Expect `59 passed` (takes 1–2 minutes).
5. Try the commands, in this order:

   ```bash
   ptaa --help
   ptaa settings                                   # every secret shows as ••••
   ptaa fees 1000 100                              # Indian fee breakdown for a 1,00,000 trade
   ptaa live --simulate --seconds 10               # fast path on fake ticks; watch p99 latency in microseconds
   ptaa train gbm --synthetic                      # trains a model on synthetic data in ~2 s and prints metrics
   ptaa models list                                # the model is registered and verified (ok = yes)
   ptaa run RELIANCE --synthetic                   # the full agent graph; ends with HOLD or BUY
   ptaa portfolio                                  # paper book, ₹10,00,000 cash
   ptaa eval --record 12                           # 12 frozen snapshots, all checks ok
   ```

**Check:** `pytest` is green and `ptaa eval` prints `12/12 snapshots pass`. Expect `HOLD` from `ptaa run --synthetic`: on random data there is no edge and the system is built to say so.

**What the folders mean:** `data/` (created now) holds your models, caches, the paper ledger and the control database. It is yours, local, and never goes to GitHub.

## Step 3 · First run on real market data (15 minutes, free)

This uses Yahoo Finance (free, 15-minute delayed). It was impossible to test from the sandbox, so this is the first time the real-data path runs.

1. `ptaa data RELIANCE --tail 5` — downloads about 400 days of daily candles and prints the last five. Expect dates up to the last trading day.
2. `ptaa features RELIANCE` — the 37 computed features for the latest bar, plus two hashes. Any number here is what the models will see.
3. `ptaa data INFY --interval 5m --days 30 --tail 5` — intraday candles (Yahoo gives about 60 days of 5-minute history).
4. `ptaa run RELIANCE --no-llm` — the agent graph on real data with the synthetic model from step 2. The decision will be HOLD; the point is that data, features, news discovery and the audit trail work. The news analyst will search DuckDuckGo and RSS feeds for Reliance articles and print how many it kept.
5. `ptaa replay <run_id>` with the id printed by the previous command — every node, its latency and hashes.

**Check:** step 1 prints real recent dates and step 4 shows `[S3] news` with at least one article. If Yahoo returns nothing, wait a minute and retry; Yahoo rate-limits bursts.

**Do not** judge the model yet; it was trained on synthetic data. Step 4 fixes that.

## Step 4 · Train the first real model on Colab (45 minutes, free GPU)

1. Open https://github.com/purushapple2024-code/PurushTradingAIAgents and click the **Open in Colab** badge in the README (it opens `training/notebooks/train_gbm_colab.ipynb`). Sign in with a Google account.
2. In Colab: menu **Runtime → Change runtime type → T4 GPU** (GBM training is fine on CPU too; the GPU matters for the LSTM later).
3. Run the cells top to bottom with Shift+Enter. The second cell asks to **mount Google Drive**; accept. Models are saved in `MyDrive/ptaa-data/models` so they survive when Colab shuts down.
4. Cell 3 downloads 12 years of NIFTY 50 daily history (2–4 minutes). Cell 4 trains with walk-forward purged cross-validation (5–10 minutes) and prints out-of-fold metrics. Cell 5 prints the fee-aware backtest next to buy-and-hold NIFTY.
5. Reading the numbers honestly: `balanced_accuracy` about 0.33 on three classes means no edge; 0.38–0.42 sustained is interesting; above 0.45 out of fold is suspicious, so tell me. The strategy row must beat the baseline row **after** fees to be worth deploying.
6. Bring the model to the laptop. Simplest: in Google Drive download the folder `ptaa-data/models` and copy its contents into your project's `data/models/` (so you have `data/models/registry.json` and `data/models/gbm/...`). Then `ptaa models list` on the laptop shows the real model with `ok = yes`.
7. Optional, later: `training/notebooks/train_seq_colab` is not yet written as a notebook; run `ptaa train seq` inside a Colab cell after installing torch, or ask me to add that notebook. `train_ppo_colab.ipynb` is there for the position sizer once the GBM is good.

**Check:** `ptaa models list` on the laptop shows a `gbm-5d-v1` trained on `NIFTY50` with today's date and `ok = yes`. Now `ptaa run RELIANCE --no-llm` uses real history and a real model.

**Repeat weekly.** The `model_freshness` risk rule refuses swing models older than 14 days by default.

## Step 5 · Install the local LLM and run the full agent graph (20 minutes)

The LLM only writes the Bull/Bear debate, the thesis narrative, the risk-persona views and the report. It never produces a number that reaches the ledger, so skipping this step changes no decision.

1. Install Ollama from https://ollama.com/download (Mac, Windows, Linux). Start it; it listens on `http://127.0.0.1:11434`.
2. Pull the model (about 2.5 GB, fits your 4 GB budget): `ollama pull qwen3:4b`. Optional fallback: `ollama pull llama3.2:3b`.
3. Confirm it answers: `ollama run qwen3:4b "say ok"` then type `/bye`.
4. In `.env` make sure these lines read:

   ```
   PTAA_LLM_BACKEND=ollama
   PTAA_LLM_MODEL=qwen3:4b
   ```
5. Run `ptaa run RELIANCE`. Now you see `BULL r1`, `BEAR r1`, `BULL r2`, `BEAR r2` lines citing `[S1]`, `[S3]`..., three persona views, and a prose report. On an 8 GB laptop a run takes 30–90 seconds because the debate makes 9 LLM calls on the CPU.
6. Run the same symbol with `--no-llm`. The decision and quantity are identical; only the prose disappears. That is the design working.

**Check:** the last line shows `llm ollama calls 9 fails 0`. If `fails` is not 0, Ollama is not running or the model name is wrong; the run still completes with templated text.

**If the laptop is too slow:** set `PTAA_LLM_BACKEND=none` for daily use and keep Ollama for reading individual runs, or use `hosted_free` with a free Groq key (data leaves your machine; your choice).

## Step 6 · Start the terminal and control plane; set up Telegram (20 minutes)

1. `ptaa serve` — starts the control-plane API at http://127.0.0.1:8765/docs and the dark terminal at http://127.0.0.1:8501. Both are local only; nobody else can reach them.
2. Open http://127.0.0.1:8501 in the browser. Tabs: **Live** (equity, positions, decisions, agent health), **Strategies** (add / edit / enable / delete, versioned), **Agents** (16 nodes, scope, model binding, permissions), **Decisions** (pick a run, see every node, the debate and the rule table), **Control** (engine on/off per module, LLM backend, risk limits, alerts, live gate, audit).
3. In **Control → Engine** tick *Swing running* and *Intraday running* and press Apply. In **Risk limits** leave the defaults for now. Every Apply creates a new version; the Revert button restores the previous one.
4. Telegram alerts (fills, rejects, halts, approval requests):
   1. In Telegram, talk to **@BotFather**, send `/newbot`, follow the prompts; copy the token it gives you (`123456:ABC...`).
   2. Start a chat with your new bot and send it any message.
   3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in the browser; find `"chat":{"id": 123456789` — that number is your chat id.
   4. Put both in `.env`: `PTAA_TELEGRAM_BOT_TOKEN=...` and `PTAA_TELEGRAM_CHAT_ID=...`. Restart `ptaa serve`.
5. Test it: `ptaa run RELIANCE` — if the decision is a BUY you get a `✅ BUY ...` message; a reject sends `⛔`. Nothing arrives for HOLD by design.

**Check:** the terminal's Control tab shows *Telegram configured: yes* and the Live tab lists your runs.

**Tip:** the terminal and the CLI share the same `data/` files, so a decision made from the CLI appears in the terminal immediately.

## Step 7 · Open broker accounts and connect live data feeds (1–3 days, waiting on KYC)

Until now the feed is Yahoo, 15 minutes delayed. Real sub-second ticks need a broker account; the API itself is free with each broker.

**Angel One (primary feed)**

1. Open a demat + trading account at https://www.angelone.in (KYC takes a day or two). You do not need to deposit money to use market data.
2. Go to https://smartapi.angelone.in, sign in, **Create an app** of type *Trading API*; note the **API key**.
3. Enable TOTP: in SmartAPI → *Enable TOTP*, scan the QR with an authenticator app **and** copy the text secret shown under the QR (you need the text, not the 6-digit code).
4. In `.env`: `PTAA_ANGEL_API_KEY`, `PTAA_ANGEL_CLIENT_CODE` (your login id), `PTAA_ANGEL_PIN`, `PTAA_ANGEL_TOTP_SECRET`.
5. `uv pip install -e ".[brokers]"` then, during market hours (9:15–15:30 IST, Monday–Friday): `ptaa live --module intraday --seconds 60`. You should see 1-minute bars printing with real prices and the status line `feed angelone`.

**Upstox (standby feed)**

1. Account at https://upstox.com; then https://account.upstox.com/developer/apps → *New app*; redirect URL `http://127.0.0.1:8765/callback`. Note API key and secret into `.env` (`PTAA_UPSTOX_API_KEY`, `PTAA_UPSTOX_API_SECRET`).
2. Upstox needs a fresh access token every day via browser login. The `ptaa auth upstox` command and the protobuf download (`--fetch-proto`) are **not written yet**; ask me when you reach this step and I will add them. Until then Angel One + Yahoo fallback is enough.

**Flattrade** is only an execution adapter (step 9); it is not needed for data.

**Check:** `ptaa settings` shows `angel_*` as `••••` (set) and `feeds available: angelone=True`; `ptaa live` prints real bars during market hours.

**Security:** these four Angel One values let anyone trade on your account. They live only in `.env` on your laptop. Never paste them in chat, screenshots or GitHub.

## Step 8 · Run paper trading daily and read the results honestly (every trading day)

Paper mode is the real product for the next several months. The routine:

1. **Morning (before 9:15 IST):** start `ptaa serve` in one terminal and, in a second terminal, `ptaa live --module intraday --trade`. The engine builds bars from the feed, runs the agent graph on each closed 5-minute bar for the top 3 candidates, paper-fills approved trades, and squares everything off at 15:10.
2. **After close (15:35):** `ptaa run <symbol>` for a few swing names or `ptaa screen --top 5` once the screen command is wired (ask me), then `ptaa portfolio` and `ptaa mark`.
3. **Weekly:** re-run the Colab GBM notebook, copy the new model to `data/models/`, check `ptaa models list` shows the new version. Run `ptaa eval` to confirm all invariants still pass.
4. **Read the terminal honestly:**
   - *Live tab*: equity vs the ₹10,00,000 start, drawdown, fees paid. Fees are the silent killer: a 1-day delivery round trip costs about 0.22%.
   - *Decisions tab*: for any BUY, read the rule table and the debate. If a rule you disagree with fired, change the limit in Control (it is versioned, you can revert).
   - Compare with buy-and-hold NIFTY over the same period. If paper equity does not beat it after fees for at least three months, the models are not good enough yet; that is information, not failure.
5. **Stop conditions the system applies by itself:** 10% drawdown halts new entries until you reset in Control; the intraday daily loss limit (2%) stops the day; the kill switch flattens and pauses everything.

**Check:** after the first full week, `ptaa portfolio` lists runs every trading day and the Telegram chat has fill/reject messages. Write down the paper equity each Friday; a plain spreadsheet is fine.

**Expectation setting:** most honest paper systems underperform at first. The value of this phase is learning which signals and rules matter on real data, with zero money at risk.

## Step 9 · Months later: what enabling live orders takes, and why to wait

I am not a financial adviser and this is not advice; it is how the gate is built. The live code exists but has never touched a real account, and SEBI treats API-driven order placement by retail investors as algorithmic trading that must go through your broker's approved route. Read your broker's current algo-trading terms before anything below.

A live order is placed only when **all five** hold, in this order:

1. `PTAA_LIVE_TRADING_ENABLED=true` is in `.env`. It cannot be switched on from the terminal, the API or the LLM.
2. In Control → Engine, execution mode is **live** (the dropdown refuses until item 1 is true) and the kill switch is off.
3. The chosen adapter (angelone / upstox / flattrade) has credentials in `.env`.
4. Today's live notional is under the cap in `settings/live.daily_notional_cap_inr` (default ₹50,000; set it to ₹5,000 for your first week).
5. You approved **that exact order** within 10 minutes: Telegram tells you `LIVE ORDER NEEDS APPROVAL · id …`; in Control → Live execution press Approve. The approval is bound to symbol, side and quantity; a changed order needs a new approval.

Suggested path when you get there: first live order with quantity 1; compare the broker contract note against the paper fee estimate; run live and paper side by side for weeks; keep the kill switch within reach. Remember that live fees in the ledger are estimates until the contract note arrives.

**Before step 9, I would want to see:** at least three months of paper results that beat buy-and-hold NIFTY after fees, no rule violations in `ptaa eval`, a stable weekly retraining habit, and a clear answer from your broker on algo approval.

## Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `ptaa: command not found` | virtual environment not active | `source .venv/bin/activate` (Windows `.venv\Scripts\activate`) in the project folder |
| `no data for RELIANCE` | Yahoo rate limit or no network | wait a minute, retry; try `--days 200` |
| `technical_analyst: no registered gbm model` | nothing in `data/models` | `ptaa train gbm --synthetic` to unblock, then step 4 for a real model |
| Every run is HOLD with `ensemble_floor` | model has no edge on current data, or synthetic model | expected until step 4; lower the floor only to study, not to trade |
| `rejected: model_freshness` | model older than 14 days (swing) / 7 days (intraday) | retrain (step 4) or raise `model_max_age_days` in Control → Risk limits |
| `llm ollama ... fails 9` | Ollama not running / wrong model name | `ollama serve`, `ollama list`; or set `PTAA_LLM_BACKEND=none` |
| `feed angelone skipped: no credentials` | Angel One keys missing in `.env` | step 7; yfinance polling is used meanwhile |
| `Angel One login failed` | wrong PIN or TOTP **secret** (you pasted the 6-digit code) | copy the text secret from the SmartAPI TOTP page |
| `database is locked` | two engines writing the same `data/` | run one `ptaa live` at a time; `ptaa serve` can stay up |
| CI red on GitHub | code or dependency problem | copy the failing step's log and send it to me |
| Terminal shows old values | browser cache | press R in the Streamlit page or reload |
| `live execution is locked` | by design | see step 9; nothing to fix |

**Where things live:** `.env` (secrets, local only) · `config.yaml` (defaults; live settings override in `data/control.db`) · `data/models` (model files + `registry.json`) · `data/ledger.db` (paper book, runs, audit) · `data/ticks` (tick archive, grows \~50 MB per day) · `data/research` (cached news pages).

**Getting help from me:** paste the command, the last 30 lines of output, and (for GitHub) the Actions log. Never paste `.env`.
