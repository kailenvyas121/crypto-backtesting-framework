"""
Verification -- the BTC prices the backtests use are real, complete, and on the 00:00 UTC day.

Uses ccxt directly (not src/), so it checks the data pipeline instead of trusting it.
Needs the network; read-only, it never writes to the cache.

  A. Integrity: timestamps are exactly 00:00 UTC and one day apart, no gaps or duplicates,
     prices positive, low <= open/close <= high, and every move over 15% listed for eyeballing.
  B. Day boundary: rebuild sample days from Coinbase's 24 HOURLY candles (00:00 to 23:00 UTC)
     and match the cached daily open/high/low/close. This proves the daily bar is 00:00 to 24:00
     UTC rather than assuming it.
  C. Source match: re-download Coinbase daily candles for the window today and compare them
     with the cache bar by bar, which catches corruption or provider revisions.
  D. Independent venue: compare daily closes with Bitstamp BTC/USD, a different exchange on the
     same 00:00 UTC day. Different order books, so small differences are expected; a big one
     would mean a bad print.
  E. Trades: every fill price in every trade ledger equals the cached close of its decision day.

Covers the full cached history (spec 3's warm-up and Study A start in 2015) through the latest closed bar.

Run from the project root:  python scripts/verify_btc_data.py
"""
import sys
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.market_data import MarketData  # noqa: E402

START = "2015-07-20"                              # first cached bar: feeds every warm-up and Study A
N_HOURLY_DAYS = 40
LEDGERS = [ROOT / "data" / "processed" / f for f in ("btc_sma_cross_50_200_trades.csv", "btc_mom_sma_trades.csv")]

day = MarketData().load("BTC/USD", start=START).set_index("timestamp")
END = str(day.index[-1].date())                    # latest closed bar in the cache
cb = ccxt.coinbase({"enableRateLimit": True})
bs = ccxt.bitstamp({"enableRateLimit": True})
fails = []


def frame(rows):
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
    return df[~df.index.duplicated()].sort_index()


def fetch_days(ex, limit, start, end):
    out, since = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ts = pd.Timestamp(end, tz="UTC")
    while True:
        rows = ex.fetch_ohlcv("BTC/USD", "1d", since=since, limit=limit)
        if not rows:
            break
        out += rows
        last = pd.Timestamp(rows[-1][0], unit="ms", tz="UTC")
        if last >= end_ts or len(rows) < 2:
            break
        since = rows[-1][0] + 86_400_000
    return frame(out).loc[start:end]


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        fails.append(name)


# ---- A. integrity --------------------------------------------------------------------------
idx = day.index
check("A1 timestamps at 00:00 UTC", bool((idx == idx.normalize()).all()) and str(idx.tz) == "UTC",
      f"{len(idx):,} bars, {idx[0].date()} -> {idx[-1].date()}")
steps = idx.to_series().diff().dropna()
check("A2 exactly one day apart, no gaps", bool((steps == pd.Timedelta(days=1)).all()) and idx.is_unique,
      f"{(steps != pd.Timedelta(days=1)).sum()} irregular steps")
ohlc_ok = (day[["open", "high", "low", "close"]] > 0).all().all() and \
          (day.low <= day[["open", "close"]].min(axis=1)).all() and (day.high >= day[["open", "close"]].max(axis=1)).all()
check("A3 prices positive, low <= open/close <= high", bool(ohlc_ok), "every bar")
ret = day.close.pct_change().dropna()
big = ret[ret.abs() > 0.15]
check("A4 large moves listed", True, f"{len(big)} days over 15%: " +
      ", ".join(f"{t.date()} {r:+.0%}" for t, r in big.items()))

# ---- B. day boundary from hourly candles ---------------------------------------------------
rng = np.random.default_rng(7)
ledger = pd.concat([pd.read_csv(f, parse_dates=["decided_at_close_of"]) for f in LEDGERS], ignore_index=True)
must = [pd.Timestamp(d, tz="UTC") for d in ledger.decided_at_close_of]          # every trade day
sample = sorted(set(must) | set(rng.choice(day.index, N_HOURLY_DAYS, replace=False)))
worst = 0.0
for d in sample:
    h = frame(cb.fetch_ohlcv("BTC/USD", "1h", since=int(d.timestamp() * 1000), limit=24))
    h = h.loc[d:d + pd.Timedelta(hours=23)]
    assert len(h) == 24, f"{d.date()}: {len(h)} hourly bars"
    rebuilt = pd.Series({"open": h.open.iloc[0], "high": h.high.max(), "low": h.low.min(), "close": h.close.iloc[-1]})
    worst = max(worst, float((rebuilt / day.loc[d, rebuilt.index] - 1).abs().max()))
check("B  daily bar = 00:00-24:00 UTC (rebuilt from hourly)", worst < 1e-9,
      f"{len(sample)} days incl. all {len(must)} trade days, worst OHLC mismatch {worst:.1e}")

# ---- C. re-download from Coinbase ----------------------------------------------------------
fresh = fetch_days(cb, 300, START, END)
both = day.index.intersection(fresh.index)
dc = float((fresh.loc[both, "close"] / day.loc[both, "close"] - 1).abs().max())
check("C  cache == fresh Coinbase download", len(both) == len(day) and dc < 1e-9,
      f"{len(both):,}/{len(day):,} bars matched, worst close mismatch {dc:.1e}")

# ---- D. independent venue ------------------------------------------------------------------
other = fetch_days(bs, 1000, START, END)
j = day.index.intersection(other.index)
gap = (other.loc[j, "close"] / day.loc[j, "close"] - 1).abs()
recent = gap.loc["2020":]
check("D1 Bitstamp closes agree 2020+ (backtest windows + warm-ups)",
      len(recent) > 0.95 * len(day.loc["2020":]) and recent.median() < 0.002 and recent.max() < 0.03,
      f"{len(recent):,} days, median gap {recent.median():.3%}, 99th pct {recent.quantile(0.99):.3%}, "
      f"max {recent.max():.3%} on {recent.idxmax().date()}")
# Before 2020 venues were less integrated (e.g. the Dec 2017 Coinbase premium). Added 2026-10-09 after the
# full-history run showed 14 such days, all in 2016-17: a gap over 3% is accepted only if Coinbase's own
# 24 hourly candles rebuild that daily close exactly, proving it is a genuine Coinbase print, not a bad bar.
early = gap.loc[:"2019"]
odd = early[early > 0.03].index
worst_rebuild = 0.0
for d in odd:
    h = frame(cb.fetch_ohlcv("BTC/USD", "1h", since=int(d.timestamp() * 1000), limit=24)).loc[d:d + pd.Timedelta(hours=23)]
    worst_rebuild = max(worst_rebuild, abs(h.close.iloc[-1] / day.loc[d, "close"] - 1))
check("D2 pre-2020 venue gaps are genuine Coinbase prints", len(early) > 0.95 * len(day.loc[:"2019"]) and worst_rebuild < 1e-9,
      f"{len(early):,} days, median gap {early.median():.3%}; {len(odd)} days over 3% "
      f"({', '.join(sorted({str(d.year) for d in odd}))}), each rebuilt from Coinbase hourly, worst {worst_rebuild:.1e}")

# ---- E. trade fills ------------------------------------------------------------------------
fill_gap = max(abs(r.trade_price_close / day.loc[pd.Timestamp(r.decided_at_close_of, tz="UTC"), "close"] - 1)
               for r in ledger.itertuples())
check("E  every trade fills at its decision-day close", fill_gap < 1e-6, f"{len(ledger)} trades, worst {fill_gap:.1e}")

print("\nall data checks passed" if not fails else f"\nFAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
