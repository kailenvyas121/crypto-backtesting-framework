"""
OHLCV validation -- FLAG, REPORT, INVESTIGATE. Nothing here modifies data.

Errors fail the dataset (it should not feed a backtest until explained):
  malformed / non-UTC timestamps, unsorted or duplicate timestamps, null OHLC,
  non-positive prices, high < low, open or close outside [low, high], negative volume.

Warnings are reported but do not fail it:
  gaps. Crypto trades 24/7, so every missing bar on the timeframe grid is a gap.
  Daily equities are checked against the XNYS session calendar, so weekends and
  exchange holidays are expected and never flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import exchange_calendars as xcals
import pandas as pd

from .crypto_data import timeframe_ms

PRICE_COLS = ["open", "high", "low", "close"]


@dataclass
class ValidationReport:
    rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing: list[pd.Timestamp] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        if self.errors:
            return "FAIL"
        return "PASS (with warnings)" if self.warnings else "PASS"

    def __str__(self) -> str:
        lines = [f"validation: {self.status}  ({self.rows:,} rows)"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        return "\n".join(lines)


def _expected_bars(df: pd.DataFrame, timeframe: str, asset_type: str) -> pd.DatetimeIndex:
    first, last = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    if asset_type == "equity":
        if timeframe != "1d":
            return pd.DatetimeIndex([], tz="UTC")   # intraday session grids not checked
        sessions = xcals.get_calendar("XNYS").sessions_in_range(first.tz_localize(None), last.tz_localize(None))
        return sessions.tz_localize("UTC")
    return pd.date_range(first, last, freq=pd.Timedelta(milliseconds=timeframe_ms(timeframe)))


def validate(df: pd.DataFrame, timeframe: str, asset_type: str) -> ValidationReport:
    r = ValidationReport(rows=len(df))
    if df.empty:
        r.errors.append("dataset is empty")
        return r

    ts = df["timestamp"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype) or str(ts.dt.tz) != "UTC":
        r.errors.append(f"timestamps must be tz-aware UTC, got {ts.dtype}")
        return r
    if ts.isna().any():
        r.errors.append(f"{ts.isna().sum()} null timestamps")
    if ts.duplicated().any():
        r.errors.append(f"{ts.duplicated().sum()} duplicate timestamps, e.g. {ts[ts.duplicated()].iloc[0]}")
    if not ts.is_monotonic_increasing:
        r.errors.append("timestamps are not sorted ascending")

    nulls = df[PRICE_COLS].isna().sum()
    for col, n in nulls[nulls > 0].items():
        r.errors.append(f"{n} null values in {col}")

    bad = (df[PRICE_COLS] <= 0).any(axis=1)
    if bad.any():
        r.errors.append(f"{bad.sum()} bars with a non-positive price, first at {ts[bad].iloc[0]}")
    for name, mask in {
        "high < low": df["high"] < df["low"],
        "open outside [low, high]": (df["open"] < df["low"]) | (df["open"] > df["high"]),
        "close outside [low, high]": (df["close"] < df["low"]) | (df["close"] > df["high"]),
        "negative volume": df["volume"] < 0,
    }.items():
        if mask.any():
            r.errors.append(f"{mask.sum()} bars with {name}, first at {ts[mask].iloc[0]}")

    if r.passed:
        expected = _expected_bars(df, timeframe, asset_type)
        r.missing = list(expected.difference(pd.DatetimeIndex(ts)))
        if r.missing:
            shown = ", ".join(str(t.date()) if timeframe == "1d" else str(t) for t in r.missing[:5])
            more = f" (+{len(r.missing) - 5} more)" if len(r.missing) > 5 else ""
            r.warnings.append(f"{len(r.missing)} missing bars: {shown}{more}")
    return r
