"""
Overfitting checks -- is spec 3 (12m mom + 200d SMA) or the Study A winner (close vs SMA 50) a fragile
peak, a few lucky trades, or a timing accident? Reported only: nothing here changes a headline number
or picks a new parameter. Window 31 Dec 2021 -> 31 Aug 2026 unless stated; held shorts; gross.

  A. Parameter neighbourhood: spec 3 over lookback {180..545} x SMA {100..300}; the winner over SMA
     {30..100}. A real effect degrades smoothly; an overfit one is an isolated peak.
  B. Start-date sensitivity: start at each month-end Dec 2021 -> Dec 2023, end fixed.
  C. Year by year vs buy & hold.
  D. Trade dependence: contribution of each holding run; result with the best one or two runs flat.
  E. Random-timing null: the same position path circularly shifted against the prices (same % short,
     same run lengths, wrong dates), 2,000 draws. Does the real timing beat shifted timing?
  F. Study A null at 10,000 shuffles: a precise p-value for the winner (the 1,000-draw cut-off moves
     with the seed: 2.66 vs 2.69).
  G. Spec 3 before 2022 (Jul 2016 -> Dec 2021, NOT out of sample: the rule was posted after it worked).

Writes data/processed/robustness_btc_mom_sma_grid.csv and prints the report.

Run from the project root:  python scripts/robustness_btc_trend.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backtest, metrics, predictability, signals  # noqa: E402
from src.market_data import MarketData  # noqa: E402

ENTRY_BAR, LAST_BAR = "2021-12-31", "2026-08-31"
GRID_OUT = ROOT / "data" / "processed" / "robustness_btc_mom_sma_grid.csv"
PPY = metrics.PPY_CRYPTO

close_all = MarketData().load("BTC/USD", start="2015-07-20", end=LAST_BAR).set_index("timestamp")["close"]
close = close_all.loc[ENTRY_BAR:]
bh = backtest.run(close, signals.buy_and_hold(close))
bh_m = metrics.summary(bh, PPY)

STRATS = {
    "12m mom + 200d SMA": signals.mom_sma_long_short(close_all, 365, 200),
    "close vs SMA 50": signals.price_vs_sma(close_all, 50),
}


def run(pos_all, c=close):
    return backtest.run(c, pos_all.loc[c.index], sizing="units")


print(f"buy & hold on the window: total {bh_m['total_return']:+.0%}  Sharpe {bh_m['sharpe']:.2f}  "
      f"max DD {bh_m['max_drawdown']:.0%}")

# ---- A. parameter neighbourhood --------------------------------------------------------------
print("\nA. spec 3 parameter neighbourhood: Sharpe (total return) on the window; headline = 365 / 200")
rows = []
for lb in (180, 270, 365, 455, 545):
    for w in (100, 150, 200, 250, 300):
        m = metrics.summary(run(signals.mom_sma_long_short(close_all, lb, w)), PPY)
        rows.append({"lookback": lb, "sma": w, "sharpe": m["sharpe"], "total_return": m["total_return"],
                     "max_drawdown": m["max_drawdown"]})
grid = pd.DataFrame(rows)
grid.to_csv(GRID_OUT, index=False)
show = grid.assign(cell=lambda d: d.sharpe.map("{:.2f}".format) + " (" + d.total_return.map("{:+.0%}".format) + ")")
print(show.pivot(index="lookback", columns="sma", values="cell").to_string())
head = grid.query("lookback == 365 and sma == 200").sharpe.iloc[0]
print(f"   {int((grid.sharpe > bh_m['sharpe']).sum())}/25 settings beat buy & hold's Sharpe; "
      f"median {grid.sharpe.median():.2f}, min {grid.sharpe.min():.2f}; headline {head:.2f} ranks "
      f"{int((grid.sharpe > head).sum()) + 1} of 25")

print("\n   winner neighbourhood (close vs SMA n, L/S): " + "   ".join(
    f"{n}: {metrics.summary(run(signals.price_vs_sma(close_all, n)), PPY)['sharpe']:.2f}"
    for n in (30, 40, 50, 60, 75, 100, 150, 200)))

# ---- B. start-date sensitivity ---------------------------------------------------------------
print("\nB. start-date sensitivity (start at each month-end Dec 2021 -> Dec 2023, end 31 Aug 2026)")
starts = pd.date_range("2021-12-31", "2023-12-31", freq="ME", tz="UTC")
for name, pos in STRATS.items():
    beat, diffs = 0, []
    for s in starts:
        c = close_all.loc[s:pd.Timestamp(LAST_BAR, tz="UTC")]
        ms = metrics.summary(run(pos, c), PPY)["sharpe"]
        mb = metrics.summary(backtest.run(c, signals.buy_and_hold(c)), PPY)["sharpe"]
        beat += ms > mb
        diffs.append(ms - mb)
    print(f"   {name:20s} beats buy & hold's Sharpe in {beat}/{len(starts)} start dates; "
          f"Sharpe edge min {min(diffs):+.2f}, median {np.median(diffs):+.2f}")

# ---- C. year by year -------------------------------------------------------------------------
print("\nC. calendar-year returns")
yearly = {"buy & hold": bh}
yearly |= {name: run(pos) for name, pos in STRATS.items()}
tab = pd.DataFrame({k: (1 + v.strategy_ret.fillna(0)).groupby(v.index.year).prod() - 1 for k, v in yearly.items()})
tab = tab.drop(index=2021)                                 # entry bar only, no returns
tab.index = [str(y) + (" (to Aug)" if y == 2026 else "") for y in tab.index]
print(tab.map("{:+.0%}".format).to_string())

# ---- D. trade dependence ---------------------------------------------------------------------
print("\nD. trade dependence: each holding run's return (held units), then the total with the best runs flat")
for name, pos in STRATS.items():
    bt = run(pos)
    r = bt.strategy_ret.fillna(0)
    run_id = bt.position.ne(bt.position.shift()).cumsum()
    runs = pd.DataFrame({"side": bt.position.groupby(run_id).first(),
                         "start": bt.index.to_series().groupby(run_id).first(),
                         "days": r.groupby(run_id).size(),
                         "ret": (1 + r).groupby(run_id).prod() - 1,
                         "bh_ret": (1 + bh.strategy_ret.fillna(0)).groupby(run_id).prod() - 1}).dropna()
    best = runs.ret.sort_values(ascending=False).index
    out = []
    for k in (1, 2):
        rr = r.copy()
        rr[run_id.isin(best[:k])] = 0.0                    # sit in cash through the k best runs
        out.append(f"without best {k}: {(1 + rr).prod() - 1:+.0%}")
    print(f"   {name}: {len(runs)} runs; best {runs.ret.max():+.0%} ({runs.loc[best[0], 'side']:+.0f} from "
          f"{runs.loc[best[0], 'start'].date()}), all runs {(1 + r).prod() - 1:+.0%}; " + "; ".join(out))
    if name.startswith("12m"):
        print(runs.assign(side=runs.side.map({1.0: "long", -1.0: "short"}), start=runs.start.dt.date,
                          ret=runs.ret.map("{:+.0%}".format),
                          bh_ret=runs.bh_ret.map("{:+.0%}".format)).to_string(index=False))
        print("   (long runs earn exactly buy & hold's return; the edge over buy & hold is the short runs)")

# ---- E. random-timing null -------------------------------------------------------------------
print("\nE. random-timing null: same position path shifted to the wrong dates (2,000 circular shifts, >= 60 days)")
rng = np.random.default_rng(3)
for name, pos in STRATS.items():
    p = pos.loc[close.index].to_numpy()
    real = metrics.summary(run(pos), PPY)["sharpe"]
    null, wiped = [], 0
    for _ in range(2000):
        k = rng.integers(60, len(p) - 60)
        b = backtest.run(close, pd.Series(np.roll(p, k), index=close.index), sizing="units")
        if (b.equity <= 0).any():                          # a held 1x short through a >2x rally: equity gone
            wiped += 1
            null.append(-np.inf)
        else:
            null.append(metrics.sharpe(b.strategy_ret.dropna(), PPY))
    null = np.array(null)
    print(f"   {name:20s} real Sharpe {real:.2f}; shifted median {np.median(null):.2f}, 95th pct "
          f"{np.quantile(null, 0.95):.2f}; p = {np.mean(null >= real):.3f}; {wiped} of 2,000 shifted paths wiped out")

# ---- F. Study A null at 10,000 shuffles ------------------------------------------------------
pred = pd.read_csv(ROOT / "data" / "processed" / "btc_trend_predictability.csv")
disc = pred.query("horizon == 1 and period == 'discovery'").set_index("feature")
feats = pd.DataFrame({f: (signals.momentum_sign(close_all, int(f[4:])) if f.startswith("mom_") else
                          signals.price_vs_sma(close_all, int(f[7:])) if f.startswith("px_sma_") else
                          signals.sma_cross_long_short(close_all, *map(int, f[6:].split("_"))) if f.startswith("cross_") else
                          signals.mom_sma_long_short(close_all, 365, 200)) for f in disc.index})
fwd = predictability.forward_return(close_all, 1)
start = feats.dropna().index[0]
mask = (fwd.index >= start) & (fwd.index.shift(1, freq="D") <= pd.Timestamp(ENTRY_BAR, tz="UTC"))
null = predictability.shuffled_max_t(feats[mask], fwd[mask], 10, n=10_000, seed=2026)
t_win = disc.t_stat.max()
p = float(np.mean(null >= t_win))
se = np.sqrt(p * (1 - p) / len(null))
print(f"\nF. Study A best-of-13 null, 10,000 shuffles: 95th pct {np.quantile(null, 0.95):.2f}; winner "
      f"{disc.t_stat.idxmax()} t {t_win:.2f}; p = {p:.3f} +/- {se:.3f}")

# ---- G. spec 3 before 2022 -------------------------------------------------------------------
print("\nG. spec 3 before 2022 (not out of sample)")
first = STRATS["12m mom + 200d SMA"].first_valid_index()
for a, b in ((first, "2018-12-31"), ("2018-12-31", "2021-12-31")):
    c = close_all.loc[pd.Timestamp(a).tz_convert("UTC") if pd.Timestamp(a).tzinfo else pd.Timestamp(a, tz="UTC"):pd.Timestamp(b, tz="UTC")]
    ms = metrics.summary(run(STRATS["12m mom + 200d SMA"], c), PPY)
    mb = metrics.summary(backtest.run(c, signals.buy_and_hold(c)), PPY)
    print(f"   {pd.Timestamp(a).date()} -> {b}: spec 3 total {ms['total_return']:+,.0%}, Sharpe {ms['sharpe']:.2f}, "
          f"max DD {ms['max_drawdown']:.0%}  |  buy & hold {mb['total_return']:+,.0%}, Sharpe {mb['sharpe']:.2f}, "
          f"max DD {mb['max_drawdown']:.0%}")
