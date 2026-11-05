"""
Builder -- Study A in README: which of 13 pre-registered trend features predict BTC's next-day return.

Discovery: decisions whose target return lands on or before 31 Dec 2021 (from the first bar with 365
days of history). Holdout: target returns 1 Jan 2022 to 31 Aug 2026, the specs 1-3 window. Features are
ranked on discovery only; the holdout is reported in that order, never re-ranked. The discovery winner
is then traded +1/-1 (held units) over the holdout window.

Writes data/processed/btc_trend_predictability.csv (one row per feature x period x horizon),
data/processed/btc_trend_predictability_null.csv (shuffled max-t draws) and
data/processed/btc_trend_winner_2022_2026-08.csv (the overlay backtest).

Run from the project root:  python scripts/build_btc_trend_predictability.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backtest, metrics, predictability, signals  # noqa: E402
from src.market_data import MarketData  # noqa: E402

SYMBOL = "BTC/USD"
HISTORY_START = "2015-07-20"
ENTRY_BAR = "2021-12-31"      # last discovery decision whose 1-day target is in 2021 is the bar before
LAST_BAR = "2026-08-31"
HORIZONS = {1: 10, 7: 14}     # forward horizon (days) -> Newey-West lags
N_SHUFFLE = 1000
OUT = ROOT / "data" / "processed" / "btc_trend_predictability.csv"
NULL_OUT = ROOT / "data" / "processed" / "btc_trend_predictability_null.csv"
WINNER_OUT = ROOT / "data" / "processed" / "btc_trend_winner_2022_2026-08.csv"

bars = MarketData().load(SYMBOL, start=HISTORY_START, end=LAST_BAR)
close = bars.set_index("timestamp")["close"]
assert str(close.index[-1].date()) == LAST_BAR, "window not fully cached"


def build_features(c: pd.Series) -> pd.DataFrame:
    cols = {}
    for n in (30, 90, 180, 365):
        cols[f"mom_{n}"] = signals.momentum_sign(c, lookback=n)
    for n in (20, 50, 100, 200):
        cols[f"px_sma_{n}"] = signals.price_vs_sma(c, window=n)
    for f, s in ((20, 50), (50, 100), (50, 200), (100, 200)):
        cols[f"cross_{f}_{s}"] = signals.sma_cross_long_short(c, fast=f, slow=s)
    cols["mom365_sma200"] = signals.mom_sma_long_short(c, lookback=365, window=200)
    return pd.DataFrame(cols)


feats = build_features(close)
assert feats.shape[1] == 13
for name in feats:                                    # every feature passes the truncation test
    signals.assert_causal(lambda c, name=name: build_features(c)[name], close)
start = feats.dropna().index[0]
ties = int((feats.loc[start:] == 0).sum().sum())
print(f"features complete from {start.date()}; exact ties (excluded): {ties}")

rows, nulls = [], {}
for h, lags in HORIZONS.items():
    fwd = predictability.forward_return(close, h)
    target_end = fwd.index.shift(h, freq="D")         # date the target return is fully realised
    is_disc = (fwd.index >= start) & (target_end <= pd.Timestamp(ENTRY_BAR, tz="UTC"))
    is_hold = (target_end > pd.Timestamp(ENTRY_BAR, tz="UTC")) & (target_end <= pd.Timestamp(LAST_BAR, tz="UTC")) \
        & (fwd.index >= pd.Timestamp(ENTRY_BAR, tz="UTC"))
    for period, mask in (("discovery", is_disc), ("holdout", is_hold)):
        y = fwd[mask]
        for name in feats:
            rows.append({"horizon": h, "period": period, "feature": name,
                         **predictability.feature_stats(feats[name][mask], y, lags)})
        if h == 1:
            nulls[period] = predictability.shuffled_max_t(feats[mask], y, lags, n=N_SHUFFLE, seed=7)

res = pd.DataFrame(rows)
disc1 = res.query("horizon == 1 and period == 'discovery'").set_index("feature")
order = disc1.t_stat.sort_values(ascending=False).index
res["discovery_rank"] = res.feature.map({f: i + 1 for i, f in enumerate(order)})
res.sort_values(["horizon", "period", "discovery_rank"]).to_csv(OUT, index=False)
pd.DataFrame(nulls).to_csv(NULL_OUT, index_label="draw")

winner = order[0]
p95 = np.quantile(nulls["discovery"], 0.95)
p_best = float((nulls["discovery"] >= disc1.t_stat[winner]).mean())

hold1 = res.query("horizon == 1 and period == 'holdout'").set_index("feature")
show = pd.DataFrame({
    "disc spread bp": disc1.spread * 1e4, "disc t": disc1.t_stat, "disc %on": disc1.pct_on * 100,
    "hold spread bp": hold1.spread * 1e4, "hold t": hold1.t_stat, "hold %on": hold1.pct_on * 100,
    "hold timing SR": hold1.timing_sharpe,
}).loc[order].round(2)
print(f"wrote {OUT.relative_to(ROOT)}, {NULL_OUT.relative_to(ROOT)}")
print("\nnext-day return spread (feature +1 minus -1), ranked on discovery only:")
print(show.to_string())
print(f"\nshuffled best-of-13 null (discovery, {N_SHUFFLE} draws): 95th pct max t = {p95:.2f}; "
      f"winner {winner} t = {disc1.t_stat[winner]:.2f}, p(best-of-13 >= this) = {p_best:.3f}")
rank_corr = disc1.t_stat.rank().corr(hold1.t_stat.rank())
print(f"rank correlation of t-stats, discovery vs holdout: {rank_corr:+.2f}")

# overlay: discovery winner traded +1/-1 with held units on the specs 1-3 window
win_close = close.loc[ENTRY_BAR:]
bt = backtest.run(win_close, feats[winner].loc[win_close.index], sizing="units")
bt.to_csv(WINNER_OUT)
m = metrics.summary(bt, metrics.PPY_CRYPTO)
print(f"\noverlay: {winner} L/S on holdout: total {m['total_return']:+.1%}  sharpe {m['sharpe']:.2f}  "
      f"max dd {m['max_drawdown']:.1%}  -> {WINNER_OUT.relative_to(ROOT)}")
