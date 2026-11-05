"""
Offline tests for signals, backtest timing and metric definitions.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest, metrics, signals  # noqa: E402


def prices(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC"), dtype=float)


def test_position_decided_at_close_t_earns_return_t_plus_1():
    close = prices([100, 110, 99, 99])
    pos = pd.Series([1.0, -1.0, 0.0, 0.0], index=close.index)
    bt = backtest.run(close, pos)
    # day 1: held pos[0]=+1 through +10%; day 2: held pos[1]=-1 through -10%; day 3: flat
    assert np.allclose(bt.strategy_ret.iloc[1:], [0.10, 0.10, 0.0])
    assert bt.equity.iloc[0] == 1.0 and np.isnan(bt.strategy_ret.iloc[0])


def test_zero_lag_is_rejected():
    close = prices([1, 2, 3])
    with pytest.raises(AssertionError, match="lookahead"):
        backtest.run(close, signals.buy_and_hold(close), lag=0)


def test_assert_causal_passes_honest_and_catches_peeking_signal():
    close = prices(np.random.default_rng(1).lognormal(0, 0.02, 400).cumprod())
    signals.assert_causal(signals.buy_and_hold, close)
    signals.assert_causal(lambda c: np.sign(c - c.rolling(50).mean()), close)

    def peeks(c):
        return np.sign(c.shift(-1) - c)       # tomorrow's close
    with pytest.raises(AssertionError, match="future data"):
        signals.assert_causal(peeks, close)


def test_metric_definitions_on_known_series():
    r = pd.Series([0.10, -0.10, 0.10, -0.10])
    assert metrics.total_return(r) == pytest.approx(1.1 * 0.9 * 1.1 * 0.9 - 1)
    assert metrics.max_drawdown(r) == pytest.approx(1.1 * 0.9 * 1.1 * 0.9 / 1.1 - 1)   # peak 1.1 -> 0.9801
    assert metrics.cagr(r, 4) == pytest.approx(metrics.total_return(r))       # 4 bars at 4/yr = 1 year
    assert metrics.sharpe(r, 365) == pytest.approx(0.0)


def test_max_drawdown_counts_a_loss_from_entry():
    assert metrics.max_drawdown(pd.Series([-0.5, 0.2])) == pytest.approx(-0.5)


def test_sma_cross_long_short_signs_warmup_and_causality():
    close = prices(np.r_[np.linspace(100, 200, 60), np.linspace(200, 50, 60)])
    pos = signals.sma_cross_long_short(close, fast=5, slow=20)
    assert pos.iloc[:19].isna().all() and pos.iloc[19:].notna().all()      # NaN until slow SMA exists
    assert pos.iloc[19:60].eq(1).all()                                     # uptrend: fast above slow -> long
    assert pos.iloc[-1] == -1                                              # downtrend: fast below slow -> short
    fast, slow = close.rolling(5).mean(), close.rolling(20).mean()
    assert np.array_equal(pos.iloc[19:], np.sign(fast - slow).iloc[19:])  # matches the definition by hand
    signals.assert_causal(signals.sma_cross_long_short, close, fast=5, slow=20)


def test_units_sizing_holds_a_short_like_a_desk_short():
    close = prices([100, 80, 120, 60, 60, 90])
    pos = pd.Series([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], index=close.index)
    units = backtest.run(close, pos, sizing="units")
    weight = backtest.run(close, pos, sizing="weight")
    # short 1x at 100 held to 60 earns exactly +40%; then long at 60 -> 90 earns +50%
    assert units.equity.iloc[3] == pytest.approx(1.40)
    assert units.equity.iloc[-1] == pytest.approx(1.40 * 1.5)
    # daily-rebalanced short over the same path: (1.2)(0.6667)(1.5) - path dependent, not +40%
    assert weight.equity.iloc[3] == pytest.approx(1.2 * (1 - 0.5) * 1.5)
    # longs are identical under both sizings
    assert np.allclose(units.strategy_ret.iloc[4:], weight.strategy_ret.iloc[4:])
    assert np.allclose(backtest.run(close, signals.buy_and_hold(close), sizing="units").equity,
                       backtest.run(close, signals.buy_and_hold(close)).equity)


def test_trend_features_signs_warmup_and_causality():
    close = prices(np.r_[np.linspace(100, 200, 60), np.linspace(200, 50, 60)])
    mom = signals.momentum_sign(close, lookback=10)
    assert mom.iloc[:10].isna().all() and mom.iloc[10:].notna().all()
    assert mom.iloc[10:60].eq(1).all() and mom.iloc[-1] == -1
    px = signals.price_vs_sma(close, window=20)
    assert px.iloc[:19].isna().all() and px.iloc[19:60].eq(1).all() and px.iloc[-1] == -1
    for fn, kw in ((signals.momentum_sign, {"lookback": 10}), (signals.price_vs_sma, {"window": 20}),
                   (signals.mom_sma_long_short, {"lookback": 10, "window": 20})):
        signals.assert_causal(fn, close, **kw)


def test_mom_sma_long_short_needs_both_negative_to_short():
    # falls then recovers: momentum still negative while price is back above its SMA -> long
    close = prices(np.r_[np.linspace(100, 50, 40), np.linspace(50, 90, 15)])
    pos = signals.mom_sma_long_short(close, lookback=30, window=10)
    mom = close / close.shift(30) - 1
    sma = close.rolling(10).mean()
    ok = mom.notna() & sma.notna()
    expected = np.where((mom < 0) & (close < sma), -1.0, 1.0)[ok]
    assert np.array_equal(pos[ok].to_numpy(), expected) and pos[~ok].isna().all()
    assert (pos[ok] == -1).any() and (pos[ok] == 1).any()
    assert ((mom < 0) & (close > sma) & ok).any()        # the "one negative only -> stay long" case occurs


def test_newey_west_t_matches_ols_with_zero_lags_and_spread():
    from src import predictability
    rng = np.random.default_rng(3)
    x = (rng.random(500) > 0.4).astype(float)
    y = 0.01 * x + rng.normal(0, 0.02, 500)
    b, t = predictability.newey_west_t(y, x, lags=0)
    assert b == pytest.approx(y[x == 1].mean() - y[x == 0].mean())     # slope on a dummy = spread
    X = np.column_stack([np.ones(500), x])
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    white = np.linalg.inv(X.T @ X) @ (X.T * resid ** 2) @ X @ np.linalg.inv(X.T @ X)
    assert t == pytest.approx(b / np.sqrt(white[1, 1]))


def test_forward_return_is_the_next_bar_and_features_never_see_it():
    from src import predictability
    close = prices([100, 110, 99, 99])
    fwd = predictability.forward_return(close, 1)
    assert np.allclose(fwd.iloc[:3], [0.10, -0.10, 0.0]) and np.isnan(fwd.iloc[-1])
