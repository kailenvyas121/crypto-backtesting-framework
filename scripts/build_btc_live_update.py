"""
Builder -- tracking update: every BTC strategy from the 31 Dec 2021 close to the LATEST closed bar in the cache.

The pre-registered results stay on the fixed 31 Dec 2021 -> 31 Aug 2026 window (README); this file only
extends the same rules, unchanged, to whatever data `update_market_data.py` has pulled since. Bars after
31 Aug 2026 are genuinely out of sample for every spec. Also prints the position each rule holds now.

Writes data/processed/btc_live_update.csv (equity per strategy per day).

Run from the project root:  python scripts/build_btc_live_update.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backtest, metrics, signals  # noqa: E402
from src.market_data import MarketData  # noqa: E402

ENTRY_BAR, PREREG_END = "2021-12-31", "2026-08-31"
OUT = ROOT / "data" / "processed" / "btc_live_update.csv"

bars = MarketData().load("BTC/USD", start="2015-07-20")
close_all = bars.set_index("timestamp")["close"]
last = close_all.index[-1]
steps = close_all.index.to_series().diff().dropna()
assert (steps == pd.Timedelta(days=1)).all(), "gap in cached history"

RULES = {  # name -> (position on full history, sizing)
    "BTC buy & hold": (signals.buy_and_hold(close_all), "weight"),
    "50/200 SMA cross L/S": (signals.sma_cross_long_short(close_all, 50, 200), "units"),
    "12m mom + 200d SMA L/S": (signals.mom_sma_long_short(close_all, 365, 200), "units"),
    "Close vs SMA 50 L/S (Study A)": (signals.price_vs_sma(close_all, 50), "units"),
}

close = close_all.loc[ENTRY_BAR:]
res, rows = {}, []
for name, (pos, sizing) in RULES.items():
    bt = backtest.run(close, pos.loc[close.index], sizing=sizing)
    res[name] = bt.equity
    pre = bt.loc[:PREREG_END]
    post = bt.loc[PREREG_END:]
    m_all = metrics.summary(bt, metrics.PPY_CRYPTO)
    now = pos.iloc[-1]
    flips = pos.loc[close.index].ne(pos.loc[close.index].shift())
    rows.append({
        "strategy": name,
        f"total to {PREREG_END}": f"{pre.equity.iloc[-1] - 1:+.1%}",
        f"since {PREREG_END}": f"{post.equity.iloc[-1] / post.equity.iloc[0] - 1:+.1%}",
        f"total to {last.date()}": f"{m_all['total_return']:+.1%}",
        "Sharpe": f"{m_all['sharpe']:.2f}",
        "max DD": f"{m_all['max_drawdown']:.1%}",
        "position now": "long" if now > 0 else "short" if now < 0 else "flat",
        "decided at close of": str(close.index[flips][-1].date()) if name != "BTC buy & hold" else str(close.index[0].date()),
    })
out = pd.DataFrame(res)
out.to_csv(OUT)

print(f"wrote {OUT.relative_to(ROOT)}  ({len(out):,} rows, {ENTRY_BAR} -> {last.date()})")
print(f"latest close {last.date()}: ${close_all.iloc[-1]:,.2f}   "
      f"BTC since {PREREG_END}: {close_all.iloc[-1] / close_all.loc[PREREG_END] - 1:+.1%}")
print("positions 'now' = decided at the latest close, held from the next bar\n")
print(pd.DataFrame(rows).set_index("strategy").to_string())
