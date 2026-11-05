"""
Signals -- functions of price history that return a TARGET POSITION per bar.

Contract every signal follows:
  - input:  a close-price Series indexed by bar timestamp (ascending, no duplicates)
  - output: a Series on the same index, the position you would want to hold AFTER
            seeing bar t's close. +1 long, -1 short, 0 flat, fractions allowed.
  - it may use only bars <= t. The backtest applies the lag (position decided at t's
    close earns bar t+1's return); signals never shift themselves.

No lookahead, ASSERTED not just assumed: assert_causal() recomputes a signal on
truncated histories and checks every value it already produced is unchanged. A signal
that peeks at the future fails, because truncation changes its past output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(close: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=close.index, name="buy_and_hold")


def sma_cross_long_short(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """+1 when SMA(fast) > SMA(slow), -1 when below, 0 on an exact tie, NaN until SMA(slow) exists."""
    assert fast < slow, "fast window must be shorter than slow window"
    gap = close.rolling(fast, min_periods=fast).mean() - close.rolling(slow, min_periods=slow).mean()
    return np.sign(gap).rename(f"sma_{fast}_{slow}_ls")


def momentum_sign(close: pd.Series, lookback: int = 365) -> pd.Series:
    """+1 when close / close `lookback` bars earlier > 1, else -1; NaN until that bar exists."""
    mom = close / close.shift(lookback) - 1
    return pd.Series(np.where(mom > 0, 1.0, -1.0), index=close.index).where(mom.notna()).rename(f"mom_{lookback}")


def price_vs_sma(close: pd.Series, window: int = 200) -> pd.Series:
    """+1 when close > SMA(window), else -1; NaN until SMA(window) exists."""
    sma = close.rolling(window, min_periods=window).mean()
    return pd.Series(np.where(close > sma, 1.0, -1.0), index=close.index).where(sma.notna()).rename(f"px_sma_{window}")


def mom_sma_long_short(close: pd.Series, lookback: int = 365, window: int = 200) -> pd.Series:
    """Spec 3: -1 when the `lookback`-bar return < 0 AND close < SMA(window), else +1; NaN until both exist."""
    mom = close / close.shift(lookback) - 1
    sma = close.rolling(window, min_periods=window).mean()
    pos = pd.Series(np.where((mom < 0) & (close < sma), -1.0, 1.0), index=close.index)
    return pos.where(mom.notna() & sma.notna()).rename(f"mom{lookback}_sma{window}_ls")


def assert_causal(signal_fn, close: pd.Series, n_checks: int = 20, seed: int = 0, **kwargs) -> None:
    full = signal_fn(close, **kwargs)
    assert full.index.equals(close.index), f"{signal_fn.__name__}: output index must match input"
    rng = np.random.default_rng(seed)
    cuts = rng.integers(len(close) // 10, len(close), size=n_checks)
    for cut in sorted(set(cuts)):
        part = signal_fn(close.iloc[:cut], **kwargs)
        diff = ~np.isclose(part.to_numpy(float), full.iloc[:cut].to_numpy(float), equal_nan=True)
        if diff.any():
            first = part.index[diff.argmax()]
            raise AssertionError(f"{signal_fn.__name__} uses future data: value at {first} changes "
                                 f"when history is cut at {close.index[cut - 1]}")
