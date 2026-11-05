"""
Backtest -- turns a target-position series into a return stream.

Conventions (these define every number in the findings log; change them only on purpose):
  - Returns are SIMPLE close-to-close: r[t] = close[t] / close[t-1] - 1.
  - Execution lag = 1 bar: the position decided at bar t's close is held from t's close
    to t+1's close, so strategy_ret[t] = position[t-1] * r[t]. This is signal.shift(1),
    applied here and only here.
  - Gross by default. cost_bps charges |change in position| x cost at each rebalance,
    so turnover is the only thing costs scale with.
  - The first row is the entry mark: the close the first position is taken at. Its
    returns are NaN (not zero, so they never dilute a metric) and equity is 1.0 there.
  - sizing="weight" (default): the position is a constant fraction of equity, re-set every
    bar. sizing="units": when the position changes, buy/sell that fraction of equity in the
    asset and then hold those UNITS until the next change, as a desk short is held. Longs are
    identical either way; a constant -1 weight is a daily-rebalanced short (inverse-ETF
    style) and decays in volatile markets, a held short does not.
"""
from __future__ import annotations

import pandas as pd


def run(close: pd.Series, position: pd.Series, lag: int = 1, cost_bps: float = 0.0,
        sizing: str = "weight") -> pd.DataFrame:
    assert lag >= 1, "lag < 1 trades on the same close the signal was computed from (lookahead)"
    assert close.index.is_monotonic_increasing and not close.index.has_duplicates, "prices must be sorted, unique"
    assert position.index.equals(close.index), "position must be on the price index"
    assert sizing in ("weight", "units"), "sizing must be 'weight' or 'units'"

    asset_ret = close.pct_change()
    held = position.shift(lag)
    turnover = held.diff().abs().fillna(held.abs())

    exposure = held
    if sizing == "units":
        # price the units were traded at = the close before the first bar of each holding run
        run_id = held.ne(held.shift()).cumsum()
        entry_px = close.shift(1).groupby(run_id).transform("first")
        x = close.shift(1) / entry_px                     # price drift since entry, as of t-1
        exposure = held * x / (1 + held * (x - 1))        # units value / equity, entering bar t
    strat_ret = exposure * asset_ret - turnover * cost_bps / 1e4

    out = pd.DataFrame({
        "close": close,
        "signal": position,
        "position": held,
        "exposure": exposure,
        "asset_ret": asset_ret,
        "strategy_ret": strat_ret,
    })
    out["equity"] = (1 + out["strategy_ret"].fillna(0.0)).cumprod()
    out = out.iloc[lag - 1:].copy()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1
    return out
