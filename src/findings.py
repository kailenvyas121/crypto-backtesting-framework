"""
Findings log -- append-only, one row per run, nulls included.

    log_run("btc_buy_and_hold", "BTC/USD", params={"start": "2022-01-01"},
            results=summary_dict, notes="baseline")

Never rewrite or dedupe this file. A killed idea logged in July is what stops it being
re-run in September.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "findings_log.csv"
HEADER = ["date_logged", "spec", "asset", "params", "results", "notes"]


def _kv(d: dict) -> str:
    def fmt(v):
        return f"{v:.5g}" if isinstance(v, float) else str(v)
    return ", ".join(f"{k}={fmt(v)}" for k, v in d.items())


def log_run(spec: str, asset: str, params: dict, results: dict, notes: str = "") -> None:
    new = not LOG.exists()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow([date.today().isoformat(), spec, asset, _kv(params), _kv(results), notes])
