"""
Builder -- BTC buy & hold, 1 Jan 2022 to 31 Aug 2026, Coinbase BTC-USD daily closes.

Reads the raw cache only (MarketData.load, no network), runs the backtest, writes
data/processed/btc_buy_hold_2022_2026-08.csv and prints the headline numbers.

Window: $1 goes in at the close of the 31 Dec 2021 bar (00:00 UTC 1 Jan 2022) and is
marked at the close of the 31 Aug 2026 bar (00:00 UTC 1 Sep 2026), so every UTC day of
2022 through August 2026 contributes exactly one return.

Run from the project root:  python scripts/build_btc_buy_hold.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backtest, metrics, signals  # noqa: E402
from src.market_data import MarketData  # noqa: E402

SYMBOL = "BTC/USD"
ENTRY_BAR = "2021-12-31"   # close of this bar = entry
LAST_BAR = "2026-08-31"    # close of this bar = final mark
OUT = ROOT / "data" / "processed" / "btc_buy_hold_2022_2026-08.csv"

bars = MarketData().load(SYMBOL, start=ENTRY_BAR, end=LAST_BAR)
close = bars.set_index("timestamp")["close"]
assert str(close.index[0].date()) == ENTRY_BAR and str(close.index[-1].date()) == LAST_BAR, "window not fully cached"

signals.assert_causal(signals.buy_and_hold, close)
bt = backtest.run(close, signals.buy_and_hold(close))
bt.to_csv(OUT)

m = metrics.summary(bt, metrics.PPY_CRYPTO)
print(f"wrote {OUT.relative_to(ROOT)}  ({len(bt):,} rows)")
print(f"entry close {close.iloc[0]:,.2f} on {ENTRY_BAR}, final close {close.iloc[-1]:,.2f} on {LAST_BAR}")
print(metrics.format_summary(m).to_string())
