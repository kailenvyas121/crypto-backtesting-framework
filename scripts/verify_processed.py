"""
Sanity check -- every processed backtest file is re-derived from the raw price cache WITHOUT src/
(plain Python loops: no pandas rolling, no src.signals, no src.backtest, no src.metrics) and must
match. Runs offline on every `python scripts/run_all.py`.

For each of buy & hold, the 50/200 cross, spec 3 (12m mom + 200d SMA) and the Study A winner (close vs SMA 50):
  1. Calendar: one row per day, 00:00 UTC, no gaps or duplicates, exact window 31 Dec 2021 -> 31 Aug 2026.
  2. Prices: every close equals the raw cache close (no adjustment, no fill).
  3. Signal: recomputed by hand from closes at or before each bar only.
  4. Lag: the position held over bar t is the signal decided at bar t-1's close, on every row.
  5. P&L: equity rebuilt bar by bar (held units, shorts held between flips) matches to 1e-10.
  6. Metrics: total return, CAGR, Sharpe, max drawdown recomputed by hand match src/metrics.py
     AND the latest row of data/processed/../findings_log.csv for that spec.

Run from the project root:  python scripts/verify_processed.py
"""
import csv
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics  # noqa: E402  (only to compare against, never to compute the reference)
from src.market_data import MarketData  # noqa: E402

ENTRY_BAR, LAST_BAR = "2021-12-31", "2026-08-31"
P = ROOT / "data" / "processed"
LOG = ROOT / "data" / "findings_log.csv"

raw = MarketData().load("BTC/USD", start="2015-07-20", end=LAST_BAR).set_index("timestamp")["close"]
px, dates = raw.tolist(), list(raw.index)
pos_of = {d: i for i, d in enumerate(dates)}


def sma(i, n):
    return sum(px[i - n + 1:i + 1]) / n if i >= n - 1 else None


def sig_buy_hold(i):
    return 1.0


def sig_cross(i):
    f, s = sma(i, 50), sma(i, 200)
    return 1.0 if f > s else -1.0 if f < s else 0.0


def sig_mom_sma(i):
    return -1.0 if (px[i] / px[i - 365] - 1 < 0 and px[i] < sma(i, 200)) else 1.0


def sig_px_sma50(i):
    return 1.0 if px[i] > sma(i, 50) else -1.0


STRATS = [  # file, hand signal, findings-log spec name
    ("btc_buy_hold_2022_2026-08.csv", sig_buy_hold, "btc_buy_and_hold_2022_2026-08"),
    ("btc_sma_cross_50_200_2022_2026-08.csv", sig_cross, "btc_sma_cross_50_200_ls_2022_2026-08"),
    ("btc_mom_sma_2022_2026-08.csv", sig_mom_sma, "btc_mom365_sma200_ls_2022_2026-08"),
    ("btc_trend_winner_2022_2026-08.csv", sig_px_sma50, "btc_px_sma50_ls_study_a_winner_2022_2026-08"),
]

logged = {}
with LOG.open() as f:
    for row in csv.DictReader(f):                      # later rows overwrite earlier: keep the latest
        logged[row["spec"]] = dict(kv.split("=") for kv in row["results"].split(", "))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


for fname, sig, spec in STRATS:
    print(f"\n{fname}")
    d = pd.read_csv(P / fname, index_col=0, parse_dates=True)
    idx = list(d.index)

    # 1. calendar
    steps_ok = all((b - a) == pd.Timedelta(days=1) for a, b in zip(idx, idx[1:]))
    check("1 calendar", steps_ok and all(t == t.normalize() for t in idx) and str(idx[0].tz) == "UTC"
          and str(idx[0].date()) == ENTRY_BAR and str(idx[-1].date()) == LAST_BAR,
          f"{len(idx):,} daily rows {idx[0].date()} -> {idx[-1].date()}")

    # 2. prices
    worst_px = max(abs(c - px[pos_of[t]]) for t, c in zip(idx, d.close))
    check("2 closes == raw cache", worst_px == 0.0, f"worst diff {worst_px}")

    # 3. signal by hand
    i0 = pos_of[idx[0]]
    hand_sig = [sig(i) for i in range(i0, i0 + len(idx))]
    check("3 signal re-derived by hand", hand_sig == d.signal.tolist())

    # 4. lag
    lag_ok = math.isnan(d.position.iloc[0]) and d.position.iloc[1:].tolist() == hand_sig[:-1]
    check("4 position[t] == signal[t-1] (1-bar lag)", lag_ok)

    # 5. equity by hand: trade at the decision close, hold units until the signal changes
    eq, equity, held, run_eq, run_px = 1.0, [1.0], None, None, None
    for k in range(1, len(idx)):
        i = i0 + k
        want = hand_sig[k - 1]
        if want != held:
            held, run_eq, run_px = want, eq, px[i - 1]
        eq = run_eq * (1 + held * (px[i] / run_px - 1))
        equity.append(eq)
    diff = max(abs(a - b) for a, b in zip(equity, d.equity))
    check("5 equity re-derived by hand", diff < 1e-10, f"max |diff| {diff:.1e}")

    # 6. metrics by hand vs src/metrics.py vs findings log
    r = [equity[k] / equity[k - 1] - 1 for k in range(1, len(equity))]
    n = len(r)
    mean = sum(r) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1))
    peak, mdd = 1.0, 0.0
    for e in equity:
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1)
    hand = {"total_return": equity[-1] - 1, "cagr": equity[-1] ** (365 / n) - 1,
            "sharpe": mean / sd * math.sqrt(365), "max_drawdown": mdd}
    lib = metrics.summary(d, metrics.PPY_CRYPTO)
    lib_ok = all(abs(hand[k] - lib[k]) < 1e-9 for k in hand)
    log_row = logged.get(spec)
    log_ok = log_row is not None and all(abs(hand[k] - float(log_row[k])) <= 5e-5 * max(1, abs(hand[k])) for k in hand)
    check("6 metrics: hand == src/metrics.py == findings log", lib_ok and log_ok,
          f"total {hand['total_return']:+.2%}  CAGR {hand['cagr']:.2%}  Sharpe {hand['sharpe']:.3f}  "
          f"max DD {hand['max_drawdown']:.2%}" + ("" if log_row else "  (spec missing from findings log)"))

print("\nall processed files verified" if not fails else f"\nFAILED: {len(fails)} checks")
sys.exit(1 if fails else 0)
