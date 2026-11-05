"""
Performance metrics -- the ONE definition of each number. Do not re-derive these inline.

Inputs are per-bar SIMPLE returns (NaN entry bar dropped). periods_per_year is 365 for
daily crypto (trades every calendar day) and 252 for daily equities.

  total_return   prod(1 + r) - 1                            geometric
  cagr           (1 + total_return) ** (ppy / n) - 1        geometric, n = number of returns
  ann_vol        std(r, ddof=1) * sqrt(ppy)
  sharpe         mean(r) / std(r) * sqrt(ppy)               arithmetic, rf = 0
  sortino        mean(r) * ppy / (sqrt(mean(min(r, 0)^2)) * sqrt(ppy))
                 downside deviation over ALL bars, target 0
  max_drawdown   min(equity / running peak - 1)
  calmar         cagr / |max_drawdown|
  trades/year    number of bars where the held position changes, / years
  % time long / short   share of bars with position > 0 / < 0

Matches the conventions behind the earlier BTC exhibit table (e.g. Calmar 0.40 / 0.77 = 0.52).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PPY_CRYPTO = 365
PPY_EQUITY = 252


def total_return(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


def cagr(r: pd.Series, ppy: int) -> float:
    return float((1 + total_return(r)) ** (ppy / len(r)) - 1)


def ann_vol(r: pd.Series, ppy: int) -> float:
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe(r: pd.Series, ppy: int) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def sortino(r: pd.Series, ppy: int) -> float:
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() * ppy / (downside * np.sqrt(ppy)))


def max_drawdown(r: pd.Series) -> float:
    equity = (1 + r).cumprod()
    return float((equity / equity.cummax().clip(lower=1.0) - 1).min())


def calmar(r: pd.Series, ppy: int) -> float:
    return cagr(r, ppy) / abs(max_drawdown(r))


def summary(bt: pd.DataFrame, ppy: int) -> dict:
    """All headline metrics for one backtest.run() frame."""
    r = bt["strategy_ret"].dropna()
    pos = bt["position"].dropna()
    years = len(r) / ppy
    changes = int((pos.diff().fillna(0) != 0).sum())
    return {
        "start": bt.index[0],
        "end": bt.index[-1],
        "bars": len(r),
        "total_return": total_return(r),
        "cagr": cagr(r, ppy),
        "ann_vol": ann_vol(r, ppy),
        "sharpe": sharpe(r, ppy),
        "sortino": sortino(r, ppy),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r, ppy),
        "trades_per_year": changes / years,
        "pct_long": float((pos > 0).mean()),
        "pct_short": float((pos < 0).mean()),
    }


def format_summary(m: dict) -> pd.Series:
    # one decimal under 1% so a small loss never prints as "-0%"
    pct = lambda x: f"{x:,.0%}" if x == 0 or abs(x) >= 0.01 else f"{x:.1%}"
    return pd.Series({
        "Total return": pct(m["total_return"]),
        "CAGR (geometric)": pct(m["cagr"]),
        "Annualized volatility": pct(m["ann_vol"]),
        "Sharpe ratio": f"{m['sharpe']:.2f}",
        "Sortino ratio": f"{m['sortino']:.2f}",
        "Max drawdown": pct(m["max_drawdown"]),
        "Calmar ratio": f"{m['calmar']:.2f}",
        "Trades / year": f"{m['trades_per_year']:.1f}",
        "% time long": pct(m["pct_long"]),
        "% time short": pct(m["pct_short"]),
    })
