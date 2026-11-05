"""
Builder -- BTC 50/200-day SMA trend, long/short, 1 Jan 2022 to 31 Aug 2026 (spec 2 in README).

Reads the raw cache only (MarketData.load, no network). The moving averages are computed on
history from WARMUP_START so the 200-day average already exists at the entry bar; the backtest
itself covers exactly the buy & hold window, so the two are directly comparable.

Shorts are HELD (sizing="units"): sell at the cross, hold those units to the next cross. The
daily-rebalanced short (sizing="weight") is printed as a sensitivity, as is a 10 bps cost.

Writes data/processed/btc_sma_cross_50_200_2022_2026-08.csv and prints the headline numbers
next to buy & hold.

Run from the project root:  python scripts/build_btc_sma_cross.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backtest, metrics, signals  # noqa: E402
from src.market_data import MarketData  # noqa: E402

SYMBOL = "BTC/USD"
WARMUP_START = "2020-01-01"   # >200 bars before entry; only feeds the moving averages
ENTRY_BAR = "2021-12-31"      # close of this bar = entry (same as buy & hold)
LAST_BAR = "2026-08-31"       # close of this bar = final mark
FAST, SLOW = 50, 200
COST_SENSITIVITY_BPS = 10
SIZING = "units"
OUT = ROOT / "data" / "processed" / "btc_sma_cross_50_200_2022_2026-08.csv"
BENCH = ROOT / "data" / "processed" / "btc_buy_hold_2022_2026-08.csv"

bars = MarketData().load(SYMBOL, start=WARMUP_START, end=LAST_BAR)
close_all = bars.set_index("timestamp")["close"]
assert str(close_all.index[-1].date()) == LAST_BAR, "window not fully cached"

signals.assert_causal(signals.sma_cross_long_short, close_all, fast=FAST, slow=SLOW)
pos_all = signals.sma_cross_long_short(close_all, fast=FAST, slow=SLOW)

window = close_all.loc[ENTRY_BAR:]
close, pos = window, pos_all.loc[window.index]
assert str(close.index[0].date()) == ENTRY_BAR
assert pos.notna().all(), "200-day average not warmed up by the entry bar"

bt = backtest.run(close, pos, sizing=SIZING)
bt.to_csv(OUT)

bench = pd.read_csv(BENCH, index_col=0, parse_dates=True)
assert bench.index.equals(bt.index), "benchmark and strategy windows differ"

net = backtest.run(close, pos, cost_bps=COST_SENSITIVITY_BPS, sizing=SIZING)
daily = backtest.run(close, pos, sizing="weight")
table = pd.DataFrame({
    "BTC buy & hold": metrics.format_summary(metrics.summary(bench, metrics.PPY_CRYPTO)),
    f"{FAST}/{SLOW} L/S (gross)": metrics.format_summary(metrics.summary(bt, metrics.PPY_CRYPTO)),
    f"{FAST}/{SLOW} L/S ({COST_SENSITIVITY_BPS} bps)": metrics.format_summary(metrics.summary(net, metrics.PPY_CRYPTO)),
    "daily-rebal short": metrics.format_summary(metrics.summary(daily, metrics.PPY_CRYPTO)),
})

flips = bt.position.diff().fillna(0).ne(0)
print(f"wrote {OUT.relative_to(ROOT)}  ({len(bt):,} rows)")
print(f"position at entry: {pos.iloc[0]:+.0f}   ({'long' if pos.iloc[0] > 0 else 'short'} going into 1 Jan 2022)")
print("crosses held in window:")
for ts, p in bt.position[flips].items():
    print(f"  {ts.date()}  -> {'long' if p > 0 else 'short'}")
print()
print(table.to_string())
