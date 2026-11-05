"""
Equity / ETF / index OHLCV through yfinance.

Design rules:
  - auto_adjust=False. Both close and adj_close are kept; research code chooses. Splits
    and dividends are kept as their own columns so adjustments stay auditable.
  - Daily bars are labelled by SESSION DATE at 00:00 UTC. yfinance hands back New York
    midnight; converting that to UTC would push every bar to 04:00/05:00 UTC, and a naive
    .date() in another zone can shift it a day. The session date is the label; the close
    it represents is the official NYSE close (16:00 ET, 13:00 on early-close days).
  - Intraday bars keep their exact timestamps, converted to UTC. Yahoo only serves recent
    intraday history (roughly 60 days for <1h, 730 days for 1h), so long-horizon equity
    research should use 1d.
  - Today's session is incomplete until the XNYS calendar says it has closed.
  - No repair=True, no filling. Yahoo occasionally drops or delays a session; validation
    reports that against the exchange calendar rather than papering over it.
"""
from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

COLUMNS = ["timestamp", "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
CALENDAR = "XNYS"
DAILY = {"1d", "5d", "1wk", "1mo", "3mo"}

_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
           "Adj Close": "adj_close", "Volume": "volume",
           "Dividends": "dividends", "Stock Splits": "splits"}


def session_close_utc(session: pd.Timestamp) -> pd.Timestamp:
    """Official close of one XNYS session, in UTC."""
    return xcals.get_calendar(CALENDAR).session_close(session.tz_localize(None).normalize())


class YFinanceProvider:
    name = "yfinance"

    def fetch(self, ticker: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp | None = None,
              include_incomplete: bool = False) -> pd.DataFrame:
        raw = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d") if end is not None else None,
            interval=timeframe,
            auto_adjust=False,
            actions=True,
            repair=False,
        )
        if raw.empty:
            return pd.DataFrame(columns=COLUMNS)

        df = raw.rename(columns=_RENAME)
        if timeframe in DAILY:
            dates = df.index.tz_localize(None).normalize() if df.index.tz else df.index.normalize()
            df.index = dates.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"
        df = df.reset_index()
        for col in COLUMNS:
            if col not in df:
                df[col] = 0.0
        df = df[COLUMNS].drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)

        if not include_incomplete and timeframe == "1d" and len(df):
            last = df["timestamp"].iloc[-1]
            cal = xcals.get_calendar(CALENDAR)
            if cal.is_session(last.tz_localize(None)) and pd.Timestamp.now(tz="UTC") < session_close_utc(last):
                df = df.iloc[:-1].reset_index(drop=True)
        return df
