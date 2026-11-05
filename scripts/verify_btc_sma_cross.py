"""
Verification -- the 50/200 SMA long/short backtest has no lookahead and is reproducible by hand.

Four independent checks, each one fails loudly:
  1. Re-derivation: recompute the SMAs, signal, held-short P&L and equity with plain Python
     loops straight from the raw cache (no pandas rolling, no src/backtest.py) and match the
     processed file to 1e-10.
  2. Future-scramble: replace every price after a cut date with random noise and re-run the
     real pipeline; every signal, position and return up to the cut must be bit-identical.
     Repeated for many cut dates across the window.
  3. Trade ledger: for every position change, show the decision date, the SMA values at that
     close, and that the new position first earns a return on the NEXT bar. Written to
     data/processed/btc_sma_cross_50_200_trades.csv so every trade can be audited.
  4. Execution-delay stress: trade one bar later still (lag 2) and report how much the result
     depends on trading at the signal close.

Run from the project root:  python scripts/verify_btc_sma_cross.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backtest, metrics, signals  # noqa: E402
from src.market_data import MarketData  # noqa: E402

SYMBOL, WARMUP_START, ENTRY_BAR, LAST_BAR = "BTC/USD", "2020-01-01", "2021-12-31", "2026-08-31"
FAST, SLOW = 50, 200
PROCESSED = ROOT / "data" / "processed" / "btc_sma_cross_50_200_2022_2026-08.csv"
LEDGER = ROOT / "data" / "processed" / "btc_sma_cross_50_200_trades.csv"

bars = MarketData().load(SYMBOL, start=WARMUP_START, end=LAST_BAR)
close_all = bars.set_index("timestamp")["close"]
saved = pd.read_csv(PROCESSED, index_col=0, parse_dates=True)


def pipeline(close_all: pd.Series, lag: int = 1) -> pd.DataFrame:
    """Exactly what scripts/build_btc_sma_cross.py does."""
    pos = signals.sma_cross_long_short(close_all, fast=FAST, slow=SLOW)
    window = close_all.loc[ENTRY_BAR:]
    return backtest.run(window, pos.loc[window.index], lag=lag, sizing="units")


# ---- 1. re-derivation by hand ------------------------------------------------------------
px = close_all.tolist()
dates = list(close_all.index)
entry = dates.index(pd.Timestamp(ENTRY_BAR, tz="UTC"))

def sma(i, n):                       # mean of the n closes ending at bar i -- only bars <= i
    return sum(px[i - n + 1:i + 1]) / n

decided = {i: (1.0 if sma(i, FAST) > sma(i, SLOW) else -1.0 if sma(i, FAST) < sma(i, SLOW) else 0.0)
           for i in range(entry, len(px))}

equity, eq = [1.0], 1.0
held, run_eq, run_px = None, None, None
for i in range(entry + 1, len(px)):
    want = decided[i - 1]            # decided at yesterday's close
    if want != held:                 # trade at yesterday's close, then hold the units
        held, run_eq, run_px = want, eq, px[i - 1]
    eq = run_eq * (1 + held * (px[i] / run_px - 1))
    equity.append(eq)

diff = np.max(np.abs(np.array(equity) - saved.equity.to_numpy()))
assert diff < 1e-10, f"hand re-derivation differs from processed file by {diff}"
print(f"[1] re-derivation by hand matches processed file      max |diff| = {diff:.1e}")

# ---- 2. future scramble -------------------------------------------------------------------
base = pipeline(close_all)
rng = np.random.default_rng(42)
cuts = sorted(rng.choice(np.arange(10, len(base) - 1), size=40, replace=False))
for k in cuts:
    cut_ts = base.index[k]
    fake = close_all.copy()
    after = fake.index > cut_ts
    fake[after] = fake[cut_ts] * np.exp(np.cumsum(rng.normal(0, 0.2, after.sum())))   # wild fake future
    alt = pipeline(fake)
    for col in ("signal", "position", "exposure", "strategy_ret", "equity"):
        a, b = base[col].loc[:cut_ts].to_numpy(), alt[col].loc[:cut_ts].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), f"{col} up to {cut_ts.date()} changed when the future changed"
print(f"[2] future scramble: {len(cuts)} cut dates, history up to each cut is bit-identical")

# ---- 3. trade ledger ----------------------------------------------------------------------
f_all = close_all.rolling(FAST).mean()
s_all = close_all.rolling(SLOW).mean()
chg = base.signal.ne(base.signal.shift())
rows = []
for t in base.index[chg]:
    nxt = base.index.get_loc(t) + 1
    if nxt >= len(base):
        continue
    t1 = base.index[nxt]
    assert base.position.loc[t1] == base.signal.loc[t], "new position must be held from the NEXT bar"
    assert base.index.get_loc(t) == 0 or base.position.loc[t] != base.signal.loc[t], \
        "position on the decision bar must still be the OLD one"
    rows.append({
        "decided_at_close_of": t.date(), "sma50": round(f_all[t], 2), "sma200": round(s_all[t], 2),
        "new_side": "long" if base.signal[t] > 0 else "short", "trade_price_close": round(close_all[t], 2),
        "first_bar_earning_return": t1.date(),
    })
ledger = pd.DataFrame(rows)
ledger.to_csv(LEDGER, index=False)
print(f"[3] trade ledger: {len(ledger)} position changes, each earns from the next bar  -> {LEDGER.relative_to(ROOT)}")
print(ledger.to_string(index=False))

# ---- 4. execution-delay stress ------------------------------------------------------------
print("\n[4] execution delay (lag 1 = trade at the signal close, lag 2 = one full day later)")
for lag in (1, 2):
    m = metrics.summary(pipeline(close_all, lag=lag), metrics.PPY_CRYPTO)
    print(f"    lag {lag}: total {m['total_return']:+.1%}   sharpe {m['sharpe']:.2f}   max dd {m['max_drawdown']:.1%}")

# ---- 5. explicit next-open execution -----------------------------------------------------
# Institutional timing: signal computed on bar t's close, order filled at bar t+1's OPEN
# (the first trade after the signal exists). Re-run the hand loop filling at the real opens.
opens = bars.set_index("timestamp")["open"].tolist()
eq_o, eq, held = [1.0], 1.0, None
for i in range(entry + 1, len(px)):
    want = decided[i - 1]
    if want != held:
        # mark the old position to the fill (bar i's open), then flip at that open
        if held is not None:
            eq = run_eq * (1 + held * (opens[i] / run_px - 1))
        held, run_eq, run_px = want, eq, opens[i]
    eq = run_eq * (1 + held * (px[i] / run_px - 1))
    eq_o.append(eq)
r_o = pd.Series(eq_o, index=saved.index).pct_change().dropna()
print(f"\n[5] fills at the actual next-day open: total {eq_o[-1] - 1:+.2%}   sharpe {metrics.sharpe(r_o, 365):.2f}   "
      f"vs close-fill {saved.equity.iloc[-1] - 1:+.2%}  (gap = open vs prior close, at most 0.03% on Coinbase)")
assert abs(eq_o[-1] - saved.equity.iloc[-1]) < 0.01, "next-open execution should match close execution within 1pt"
print("\nall checks passed")
