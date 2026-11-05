"""
Builder -- BTC 12-month momentum + 200-day SMA, long/short, 1 Jan 2022 to 31 Aug 2026 (spec 3 in README).

Short when the 365-day return < 0 AND close < SMA(200); otherwise long. Same window, sizing and
sensitivities as spec 2 (scripts/build_btc_sma_cross.py), so all three are directly comparable.
The secondary run over the full cached history (from the first bar with 365 days behind it) is
reported only, not judged on.

Writes data/processed/btc_mom_sma_2022_2026-08.csv and prints the headline numbers next to buy & hold
and the 50/200 cross.

Run from the project root:  python scripts/build_btc_mom_sma.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backtest, metrics, signals  # noqa: E402
from src.market_data import MarketData  # noqa: E402

SYMBOL = "BTC/USD"
HISTORY_START = "2015-07-20"  # first cached bar; feeds the warm-up and the secondary full-history run
ENTRY_BAR = "2021-12-31"      # close of this bar = entry (same as buy & hold)
LAST_BAR = "2026-08-31"       # close of this bar = final mark
LOOKBACK, WINDOW = 365, 200
COST_SENSITIVITY_BPS = 10
SIZING = "units"
OUT = ROOT / "data" / "processed" / "btc_mom_sma_2022_2026-08.csv"
BENCH = ROOT / "data" / "processed" / "btc_buy_hold_2022_2026-08.csv"
SMA_CROSS = ROOT / "data" / "processed" / "btc_sma_cross_50_200_2022_2026-08.csv"

bars = MarketData().load(SYMBOL, start=HISTORY_START, end=LAST_BAR)
close_all = bars.set_index("timestamp")["close"]
assert str(close_all.index[-1].date()) == LAST_BAR, "window not fully cached"

signals.assert_causal(signals.mom_sma_long_short, close_all, lookback=LOOKBACK, window=WINDOW)
pos_all = signals.mom_sma_long_short(close_all, lookback=LOOKBACK, window=WINDOW)

window = close_all.loc[ENTRY_BAR:]
close, pos = window, pos_all.loc[window.index]
assert str(close.index[0].date()) == ENTRY_BAR
assert pos.notna().all(), "momentum / SMA not warmed up by the entry bar"

bt = backtest.run(close, pos, sizing=SIZING)
bt.to_csv(OUT)

bench = pd.read_csv(BENCH, index_col=0, parse_dates=True)
cross = pd.read_csv(SMA_CROSS, index_col=0, parse_dates=True)
assert bench.index.equals(bt.index) and cross.index.equals(bt.index), "windows differ"

net = backtest.run(close, pos, cost_bps=COST_SENSITIVITY_BPS, sizing=SIZING)
daily = backtest.run(close, pos, sizing="weight")
fmt = lambda b: metrics.format_summary(metrics.summary(b, metrics.PPY_CRYPTO))
table = pd.DataFrame({
    "BTC buy & hold": fmt(bench),
    "50/200 L/S": fmt(cross),
    "mom+SMA L/S (gross)": fmt(bt),
    f"mom+SMA ({COST_SENSITIVITY_BPS} bps)": fmt(net),
    "daily-rebal short": fmt(daily),
})

# secondary, reported only: full cached history from the first bar the rule exists
first = pos_all.first_valid_index()
full_close = close_all.loc[first:]
full = pd.DataFrame({
    "BTC buy & hold": fmt(backtest.run(full_close, signals.buy_and_hold(full_close))),
    "mom+SMA L/S": fmt(backtest.run(full_close, pos_all.loc[first:], sizing=SIZING)),
})

flips = bt.position.diff().fillna(0).ne(0)
print(f"wrote {OUT.relative_to(ROOT)}  ({len(bt):,} rows)")
print(f"position at entry: {pos.iloc[0]:+.0f}   ({'long' if pos.iloc[0] > 0 else 'short'} going into 1 Jan 2022)")
print("position changes held in window:")
for ts, p in bt.position[flips].items():
    print(f"  {ts.date()}  -> {'long' if p > 0 else 'short'}")
print()
print(table.to_string())
print(f"\nsecondary (not judged on): full history, entry close {first.date()} to {LAST_BAR}")
print(full.to_string())
