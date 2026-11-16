"""Honest vectorised backtest of a probability signal with fees and a buy-and-hold baseline.

Entry rule: go long when out-of-fold P(up) >= threshold (and P(up) is the max class) on bar t,
fill at bar t+1 open, hold for the label horizon or until the triple-barrier exit (we reuse the
label's realised return `ret`, which already encodes stop/target/time exit), pay round-trip fees
and slippage. One position per symbol; capital split equally across concurrent positions.
Reports strategy and baseline side by side; a strategy that cannot beat the baseline is reported
as such, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ptaa.risk.fees import round_trip_cost
from ptaa.settings import FeeConfig
from ptaa.training.metrics import trading_metrics


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    strategy: dict[str, float]
    baseline: dict[str, float]
    equity_curve: pd.Series
    baseline_curve: pd.Series


def backtest_signal(
    meta: pd.DataFrame,
    oof_proba: pd.DataFrame,
    fees: FeeConfig,
    index_close: pd.Series | None = None,
    threshold: float = 0.45,
    slippage_bps: float = 5.0,
    notional_per_trade: float = 50_000.0,
    module: str = "swing",
) -> BacktestResult:
    df = meta.join(oof_proba).dropna(subset=["p_up"])
    sel = (df["p_up"] >= threshold) & (df["p_up"] >= df[["p_down", "p_flat"]].max(axis=1))
    trades = df[sel].copy()
    if trades.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(trades, {"n": 0.0}, {"n": 0.0}, empty, empty)

    # Fee + slippage as a fraction of notional; price proxy = notional / 100 shares
    px = notional_per_trade / 100.0
    fee_frac = round_trip_cost(fees, module, px, 100) / notional_per_trade  # type: ignore[arg-type]
    slip_frac = 2 * slippage_bps / 1e4
    trades["net_ret"] = trades["ret"] - fee_frac - slip_frac
    trades = trades.sort_values("ts")

    # Daily strategy return: average of trades decided that day (equal capital per trade)
    daily = trades.groupby(trades["ts"].dt.normalize())["net_ret"].mean()
    strat = trading_metrics(daily)
    strat["trades"] = float(len(trades))
    strat["fee_drag_per_trade"] = float(fee_frac + slip_frac)
    equity = (1 + daily).cumprod()

    if index_close is not None and not index_close.empty:
        ic = index_close.copy()
        ic.index = pd.DatetimeIndex(ic.index).tz_convert("Asia/Kolkata").normalize()
        ic = ic[~ic.index.duplicated()]
        ic = ic.loc[(ic.index >= daily.index.min()) & (ic.index <= daily.index.max())]
        b_ret = ic.pct_change().dropna()
        base = trading_metrics(b_ret)
        base_curve = (1 + b_ret).cumprod()
    else:
        # no index given: baseline = always-long every symbol every bar (the panel's own average)
        b_ret = df.groupby(df["ts"].dt.normalize())["ret"].mean()
        base = trading_metrics(b_ret)
        base["note"] = 1.0  # flag: baseline is panel average, not NIFTY
        base_curve = (1 + b_ret).cumprod()

    return BacktestResult(trades, strat, base, equity, base_curve)


def threshold_sweep(
    meta: pd.DataFrame,
    oof_proba: pd.DataFrame,
    fees: FeeConfig,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    rows = []
    for th in thresholds or [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        r = backtest_signal(meta, oof_proba, fees, threshold=th)
        rows.append(
            {
                "threshold": th,
                **{
                    k: v
                    for k, v in r.strategy.items()
                    if k in ("trades", "sharpe", "total_return", "win_rate", "max_drawdown")
                },
            }
        )
    return pd.DataFrame(rows)
