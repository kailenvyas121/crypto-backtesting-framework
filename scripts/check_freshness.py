"""
Staleness check for final charts -- by content hash, not mtime.

A chart is stale when any declared input has changed since it was last rendered.
mtimes do not work for this (a bulk touch makes everything look stale), so each chart's
inputs are hashed and compared with charts/freshness_manifest.json.

    python scripts/check_freshness.py            # verify
    python scripts/check_freshness.py --update   # re-record after re-rendering

The one manual step: when a chart gains an input, add it to CHARTS.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "charts" / "freshness_manifest.json"

BTC_BH = "data/processed/btc_buy_hold_2022_2026-08.csv"
BTC_SMA = "data/processed/btc_sma_cross_50_200_2022_2026-08.csv"
BTC_MS = "data/processed/btc_mom_sma_2022_2026-08.csv"
BTC_WIN = "data/processed/btc_trend_winner_2022_2026-08.csv"
PRED = "data/processed/btc_trend_predictability.csv"
PRED_NULL = "data/processed/btc_trend_predictability_null.csv"
STYLE = "scripts/house_style.py"

CHARTS = {
    "charts/btc_buy_hold/final/btc_buy_hold_growth.png":
        ["scripts/chart_btc_buy_hold_growth.py", STYLE, BTC_BH],
    "charts/btc_buy_hold/final/btc_buy_hold_drawdown.png":
        ["scripts/chart_btc_buy_hold_drawdown.py", STYLE, BTC_BH],
    "charts/btc_buy_hold/final/btc_buy_hold_table.png":
        ["scripts/chart_btc_buy_hold_table.py", STYLE, "src/metrics.py", BTC_BH],
    "charts/btc_sma_cross_50_200/final/btc_sma_cross_growth.png":
        ["scripts/chart_btc_sma_cross_growth.py", STYLE, BTC_SMA, BTC_BH],
    "charts/btc_sma_cross_50_200/final/btc_sma_cross_drawdown.png":
        ["scripts/chart_btc_sma_cross_drawdown.py", STYLE, BTC_SMA, BTC_BH],
    "charts/btc_sma_cross_50_200/final/btc_sma_cross_table.png":
        ["scripts/chart_btc_sma_cross_table.py", STYLE, "src/metrics.py", BTC_SMA, BTC_BH],
    "charts/btc_mom_sma/final/btc_mom_sma_growth.png":
        ["scripts/chart_btc_mom_sma_growth.py", STYLE, BTC_MS, BTC_BH],
    "charts/btc_mom_sma/final/btc_mom_sma_drawdown.png":
        ["scripts/chart_btc_mom_sma_drawdown.py", STYLE, BTC_MS, BTC_BH],
    "charts/btc_mom_sma/final/btc_mom_sma_table.png":
        ["scripts/chart_btc_mom_sma_table.py", STYLE, "src/metrics.py", BTC_MS, BTC_BH],
    "charts/btc_trend_predictability/final/btc_trend_predictability_tstats.png":
        ["scripts/chart_btc_trend_predictability.py", STYLE, PRED, PRED_NULL],
    "charts/btc_trend_predictability/final/btc_trend_overlay_growth.png":
        ["scripts/chart_btc_trend_overlay.py", STYLE, BTC_BH, BTC_SMA, BTC_WIN, BTC_MS],
    "charts/btc_live_update/final/btc_live_update_growth.png":
        ["scripts/chart_btc_live_update.py", STYLE, "data/processed/btc_live_update.csv"],
}


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()[:16]


def current() -> dict:
    return {chart: {"output": sha(chart), "inputs": {p: sha(p) for p in deps}}
            for chart, deps in CHARTS.items() if (ROOT / chart).exists()}


def main() -> None:
    now = current()
    if "--update" in sys.argv:
        MANIFEST.write_text(json.dumps(now, indent=2) + "\n")
        print(f"recorded {len(now)} charts in {MANIFEST.relative_to(ROOT)}")
        return

    recorded = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    stale = 0
    for chart in CHARTS:
        if chart not in now:
            print(f"MISSING  {chart}")
            stale += 1
            continue
        old = recorded.get(chart)
        if old is None:
            print(f"UNTRACKED {chart}  (run with --update after rendering)")
            stale += 1
            continue
        changed = [p for p, h in now[chart]["inputs"].items() if old["inputs"].get(p) != h]
        if changed:
            print(f"STALE    {chart}\n         changed: {', '.join(changed)}")
            stale += 1
        else:
            print(f"fresh    {chart}")
    sys.exit(1 if stale else 0)


if __name__ == "__main__":
    main()
