"""
Predictability -- does a +1/-1 feature decided at bar t's close tell you anything about r[t+1]?

  spread    mean(fwd | feature = +1) - mean(fwd | feature = -1)
  t-stat    Newey-West (Bartlett kernel) t of the slope in fwd = a + b * 1[feature = +1], i.e. of the
            spread, robust to autocorrelation and heteroskedasticity. Overlapping multi-day targets
            need lags >= horizon.
  hit rate  share of bars where sign(fwd) == feature
  timing Sharpe   Sharpe of feature[t] * r[t+1], the daily-rebalanced +1/-1 return

forward_return() is the only place the future is touched: it aligns r[t+1..t+h] onto bar t on
purpose, as the TARGET. Features must never be built from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Simple return from bar t's close to bar t+horizon's close, stored at bar t (NaN at the end)."""
    return (close.shift(-horizon) / close - 1).rename(f"fwd_{horizon}")


def newey_west_t(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[float, float]:
    """OLS slope of y on [1, x] and its Newey-West t-stat."""
    X = np.column_stack([np.ones_like(x), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = (y - X @ beta)[:, None] * X
    s = u.T @ u
    for k in range(1, lags + 1):
        g = u[k:].T @ u[:-k]
        s += (1 - k / (lags + 1)) * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    return float(beta[1]), float(beta[1] / np.sqrt(cov[1, 1]))


def feature_stats(feature: pd.Series, fwd: pd.Series, lags: int, ppy: int = 365) -> dict:
    """Stats for one +1/-1 feature against a forward return, on bars where both exist."""
    df = pd.concat([feature, fwd], axis=1, keys=["f", "y"]).dropna()
    df = df[df.f != 0]
    on = (df.f > 0).to_numpy(float)
    spread, t = newey_west_t(df.y.to_numpy(), on, lags)
    timing = df.f * df.y
    return {
        "bars": len(df),
        "pct_on": float(on.mean()),
        "mean_on": float(df.y[df.f > 0].mean()),
        "mean_off": float(df.y[df.f < 0].mean()),
        "spread": spread,
        "t_stat": t,
        "hit_rate": float((np.sign(df.y) == df.f).mean()),
        "timing_sharpe": float(timing.mean() / timing.std(ddof=1) * np.sqrt(ppy)),
    }


def shuffled_max_t(features: pd.DataFrame, fwd: pd.Series, lags: int, n: int = 1000,
                   min_shift: int = 365, seed: int = 0) -> np.ndarray:
    """Null for 'best of N features': circularly shift the target against the features (keeps each
    series' own autocorrelation, breaks any link between them) and record the max t across features."""
    df = pd.concat([features, fwd.rename("_y")], axis=1).dropna()
    y = df.pop("_y").to_numpy()
    F = (df.to_numpy() > 0).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        ys = np.roll(y, rng.integers(min_shift, len(y) - min_shift))
        out[i] = max(newey_west_t(ys, F[:, j], lags)[1] for j in range(F.shape[1]))
    return out
