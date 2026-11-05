"""
One command: rebuild every backtest, audit it, and redraw every final chart. Stops at the first failure.

    python scripts/run_all.py              # offline: uses cached prices
    python scripts/run_all.py --online     # also refreshes prices and runs the data audit (needs network)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("offline tests",                 ["-m", "pytest", "tests", "-q"]),
    ("build buy & hold",              ["scripts/build_btc_buy_hold.py"]),
    ("build 50/200 long/short",       ["scripts/build_btc_sma_cross.py"]),
    ("audit backtest (no lookahead)", ["scripts/verify_btc_sma_cross.py"]),
    ("build 12m mom + 200d SMA L/S",  ["scripts/build_btc_mom_sma.py"]),
    ("audit 12m mom + 200d SMA",      ["scripts/verify_btc_mom_sma.py"]),
    ("build trend predictability study", ["scripts/build_btc_trend_predictability.py"]),
    ("audit trend study (no leakage)", ["scripts/verify_btc_trend_predictability.py"]),
    ("sanity: re-derive every backtest by hand", ["scripts/verify_processed.py"]),
    ("overfitting checks",            ["scripts/robustness_btc_trend.py"]),
    ("build tracking update (to latest bar)", ["scripts/build_btc_live_update.py"]),
    ("chart buy & hold",              ["scripts/chart_btc_buy_hold_growth.py"]),
    ("",                              ["scripts/chart_btc_buy_hold_drawdown.py"]),
    ("",                              ["scripts/chart_btc_buy_hold_table.py"]),
    ("chart 50/200 vs buy & hold",    ["scripts/chart_btc_sma_cross_growth.py"]),
    ("",                              ["scripts/chart_btc_sma_cross_drawdown.py"]),
    ("",                              ["scripts/chart_btc_sma_cross_table.py"]),
    ("chart 12m mom + 200d SMA vs buy & hold", ["scripts/chart_btc_mom_sma_growth.py"]),
    ("",                              ["scripts/chart_btc_mom_sma_drawdown.py"]),
    ("",                              ["scripts/chart_btc_mom_sma_table.py"]),
    ("chart trend predictability + overlay", ["scripts/chart_btc_trend_predictability.py"]),
    ("",                              ["scripts/chart_btc_trend_overlay.py"]),
    ("chart tracking update",         ["scripts/chart_btc_live_update.py"]),
    ("record chart freshness",        ["scripts/check_freshness.py", "--update"]),
]
ONLINE = [
    ("refresh prices",                ["scripts/update_market_data.py"]),
    ("audit prices (00:00 UTC, vs Coinbase + Bitstamp)", ["scripts/verify_btc_data.py"]),
]

steps = (ONLINE if "--online" in sys.argv else []) + STEPS
for name, args in steps:
    if name:
        print(f"\n=== {name}")
    res = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    tail = (res.stdout + res.stderr).strip().splitlines()
    print("\n".join(tail[-3:]))
    if res.returncode != 0:
        print(f"\nSTOPPED: {' '.join(args)} failed (exit {res.returncode})\n" + "\n".join(tail[-25:]))
        sys.exit(res.returncode)
print("\nall steps passed")
