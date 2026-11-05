"""
Verification -- Study A (trend-feature predictability) has no lookahead, no discovery/holdout leakage,
and its statistics are reproducible by hand.

  1. Split: discovery and holdout use exactly the bars the README says (counted independently), and no
     discovery target return is realised after 31 Dec 2021.
  2. Holdout scramble: replace EVERY price after 31 Dec 2021 with random noise and re-run the pipeline;
     the discovery table and ranking must be bit-identical, so 2022+ data cannot have chosen the winner.
  3. Hand re-derivation: features, forward returns, spreads, hit rates and Newey-West t-stats for all
     13 features x both periods, in plain Python loops (no pandas rolling, no numpy linear algebra, no
     src/predictability.py), matched to the processed table.
  4. Calibration: a feature that peeks at tomorrow must score an enormous t (the test can see leakage);
     200 coin-flip features must give t-stats near N(0, 1) (the t-stat is not inflated).
  5. Delay stress: every feature decided one day later (lag 2): does the discovery ranking survive?
  6. Null stability: the shuffled best-of-13 95th percentile under a different seed.

Run from the project root:  python scripts/verify_btc_trend_predictability.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import predictability, signals  # noqa: E402
from src.market_data import MarketData  # noqa: E402

ENTRY_BAR, LAST_BAR = "2021-12-31", "2026-08-31"
LAGS = 10
TABLE = ROOT / "data" / "processed" / "btc_trend_predictability.csv"
NULL = ROOT / "data" / "processed" / "btc_trend_predictability_null.csv"
E, L = pd.Timestamp(ENTRY_BAR, tz="UTC"), pd.Timestamp(LAST_BAR, tz="UTC")

close = MarketData().load("BTC/USD", start="2015-07-20", end=LAST_BAR).set_index("timestamp")["close"]
saved = pd.read_csv(TABLE)
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def features(c):
    """Exactly what scripts/build_btc_trend_predictability.py builds."""
    cols = {f"mom_{n}": signals.momentum_sign(c, lookback=n) for n in (30, 90, 180, 365)}
    cols |= {f"px_sma_{n}": signals.price_vs_sma(c, window=n) for n in (20, 50, 100, 200)}
    cols |= {f"cross_{f}_{s}": signals.sma_cross_long_short(c, fast=f, slow=s)
             for f, s in ((20, 50), (50, 100), (50, 200), (100, 200))}
    cols["mom365_sma200"] = signals.mom_sma_long_short(c, lookback=365, window=200)
    return pd.DataFrame(cols)


def discovery_table(c, lag_extra=0):
    """Discovery-period 1-day stats, as the builder computes them."""
    f = features(c).shift(lag_extra)
    start = features(c).dropna().index[0]
    fwd = predictability.forward_return(c, 1)
    mask = (fwd.index >= start) & (fwd.index.shift(1, freq="D") <= E)
    return pd.DataFrame({n: predictability.feature_stats(f[n][mask], fwd[mask], LAGS) for n in f}).T


# ---- 1. split ----------------------------------------------------------------------------
start = features(close).dropna().index[0]
n_disc_1 = len(pd.date_range(start, E - pd.Timedelta(days=1), freq="D"))   # decisions whose t+1 is <= 31 Dec 2021
n_hold_1 = len(pd.date_range(E, L - pd.Timedelta(days=1), freq="D"))       # decisions 31 Dec 2021 .. 30 Aug 2026
n_disc_7 = len(pd.date_range(start, E - pd.Timedelta(days=7), freq="D"))
n_hold_7 = len(pd.date_range(E, L - pd.Timedelta(days=7), freq="D"))
got = saved.groupby(["horizon", "period"]).bars.agg(["min", "max"])
want = {(1, "discovery"): n_disc_1, (1, "holdout"): n_hold_1, (7, "discovery"): n_disc_7, (7, "holdout"): n_hold_7}
ok = all(got.loc[k, "min"] == v and got.loc[k, "max"] == v for k, v in want.items())
check("1 split: bar counts match the README windows exactly", ok,
      f"discovery {start.date()} -> {(E - pd.Timedelta(days=1)).date()} ({n_disc_1:,} decisions, last target {E.date()}); "
      f"holdout {E.date()} -> {(L - pd.Timedelta(days=1)).date()} ({n_hold_1:,})")

# ---- 2. holdout scramble -----------------------------------------------------------------
base = discovery_table(close)
rng = np.random.default_rng(11)
for k in range(5):
    fake = close.copy()
    after = fake.index > E
    fake[after] = fake[E] * np.exp(np.cumsum(rng.normal(0, 0.1, after.sum())))
    alt = discovery_table(fake)
    same = alt.equals(base)
    if not same:
        break
order_saved = saved.query("horizon == 1 and period == 'discovery'").sort_values("discovery_rank").feature.tolist()
check("2 holdout scramble: discovery table + ranking bit-identical (5 random futures)",
      same and base.t_stat.sort_values(ascending=False).index.tolist() == order_saved)

# ---- 3. hand re-derivation ---------------------------------------------------------------
px, dates = close.tolist(), list(close.index)


def sma(i, n):
    return sum(px[i - n + 1:i + 1]) / n


def hand_feature(name, i):
    if name.startswith("mom_"):
        n = int(name[4:])
        return 1.0 if px[i] / px[i - n] - 1 > 0 else -1.0
    if name.startswith("px_sma_"):
        return 1.0 if px[i] > sma(i, int(name[7:])) else -1.0
    if name.startswith("cross_"):
        f, s = map(int, name[6:].split("_"))
        a, b = sma(i, f), sma(i, s)
        return 1.0 if a > b else -1.0 if a < b else 0.0
    return -1.0 if (px[i] / px[i - 365] - 1 < 0 and px[i] < sma(i, 200)) else 1.0


def hand_stats(xs, ys, lags):
    """Spread, hit rate and Newey-West t of y on [1, 1(x>0)], all in plain Python."""
    on = [1.0 if x > 0 else 0.0 for x in xs]
    n = len(ys)
    m_on = sum(y for y, o in zip(ys, on) if o) / sum(on)
    m_off = sum(y for y, o in zip(ys, on) if not o) / (n - sum(on))
    b = m_on - m_off
    a = m_off
    u = [(y - a - b * o) for y, o in zip(ys, on)]
    sxx = [[n, sum(on)], [sum(on), sum(on)]]                        # X'X for X = [1, on]
    det = sxx[0][0] * sxx[1][1] - sxx[0][1] * sxx[1][0]
    inv = [[sxx[1][1] / det, -sxx[0][1] / det], [-sxx[1][0] / det, sxx[0][0] / det]]
    g = [[ui, ui * o] for ui, o in zip(u, on)]                     # score per bar
    S = [[sum(gi[r] * gi[c] for gi in g) for c in range(2)] for r in range(2)]
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        for r in range(2):
            for c in range(2):
                S[r][c] += w * sum(g[t][r] * g[t - k][c] + g[t - k][r] * g[t][c] for t in range(k, n))
    # var(b) = (inv S inv)[1][1]
    row = [sum(inv[1][j] * S[j][c] for j in range(2)) for c in range(2)]
    var_b = sum(row[c] * inv[c][1] for c in range(2))
    hit = sum(1 for x, y in zip(xs, ys) if (y > 0) - (y < 0) == x) / n
    return b, b / math.sqrt(var_b), hit


i_start = dates.index(start)
i_E, i_L = dates.index(E), dates.index(L)
worst = {"spread": 0.0, "t_stat": 0.0, "hit_rate": 0.0}
for period, (lo, hi) in {"discovery": (i_start, i_E - 1), "holdout": (i_E, i_L - 1)}.items():
    rows = saved.query("horizon == 1 and period == @period").set_index("feature")
    for name in rows.index:
        xs = [hand_feature(name, i) for i in range(lo, hi + 1)]
        ys = [px[i + 1] / px[i] - 1 for i in range(lo, hi + 1)]
        keep = [(x, y) for x, y in zip(xs, ys) if x != 0]
        b, t, hit = hand_stats([k[0] for k in keep], [k[1] for k in keep], LAGS)
        worst["spread"] = max(worst["spread"], abs(b - rows.spread[name]))
        worst["t_stat"] = max(worst["t_stat"], abs(t - rows.t_stat[name]))
        worst["hit_rate"] = max(worst["hit_rate"], abs(hit - rows.hit_rate[name]))
check("3 hand re-derivation, 13 features x 2 periods", all(v < 1e-9 for v in worst.values()),
      ", ".join(f"worst {k} diff {v:.1e}" for k, v in worst.items()))

# ---- 4. calibration ----------------------------------------------------------------------
fwd = predictability.forward_return(close, 1)
mask = (fwd.index >= start) & (fwd.index.shift(1, freq="D") <= E)
peek = np.sign(fwd).replace(0, -1.0)                                   # knows tomorrow: must be caught
t_peek = predictability.feature_stats(peek[mask], fwd[mask], LAGS)["t_stat"]
rng = np.random.default_rng(5)
ts = [predictability.feature_stats(pd.Series(rng.choice([-1.0, 1.0], mask.sum()), index=fwd[mask].index),
                                   fwd[mask], LAGS)["t_stat"] for _ in range(200)]
sd, share = float(np.std(ts)), float(np.mean(np.abs(ts) > 1.96))
check("4a peeking feature is caught", t_peek > 20, f"t = {t_peek:.1f} (honest features: max 2.67)")
check("4b coin-flip features give t ~ N(0,1)", 0.8 < sd < 1.2 and share < 0.10,
      f"200 random features: sd(t) = {sd:.2f}, |t| > 1.96 in {share:.1%} (expect ~5%)")

# ---- 5. delay stress ---------------------------------------------------------------------
lag2 = discovery_table(close, lag_extra=1)
rank_corr = lag2.t_stat.rank().corr(base.t_stat.rank())
w = base.t_stat.idxmax()
print(f"INFO  5 delay stress (features one day staler): discovery winner {w} t {base.t_stat[w]:.2f} -> "
      f"{lag2.t_stat[w]:.2f}; lag-2 winner {lag2.t_stat.idxmax()}; rank corr {rank_corr:+.2f}")

# ---- 6. null stability -------------------------------------------------------------------
f_all = features(close)
alt_null = predictability.shuffled_max_t(f_all[mask], fwd[mask], LAGS, n=1000, seed=99)
p_saved = pd.read_csv(NULL).discovery.quantile(0.95)
p_alt = float(np.quantile(alt_null, 0.95))
check("6 shuffled null stable across seeds", abs(p_alt - p_saved) < 0.15,
      f"95th pct of best-of-13 t: seed 7 {p_saved:.2f}, seed 99 {p_alt:.2f}; "
      f"winner t {base.t_stat.max():.2f} beats seed-99 null: {base.t_stat.max() > p_alt}")

print("\nall checks passed" if not fails else f"\nFAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
