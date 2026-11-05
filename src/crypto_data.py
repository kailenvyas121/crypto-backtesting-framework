"""
Crypto OHLCV through CCXT -- public endpoints only, no API keys.

Design rules:
  - One exchange instance per provider, reused. enableRateLimit=True always.
  - Pagination is explicit and deterministic (see fetch()). CCXT's own auto-pagination
    is not used, so every request this makes can be read off the loop below.
  - Timestamps are bar OPEN times in UTC. A 1d bar stamped 2026-08-31 00:00 covers
    2026-08-31 00:00 -> 2026-09-01 00:00 UTC and its close is the 24:00 UTC print.
  - Only CLOSED bars are returned unless include_incomplete=True. A bar is closed when
    open_time + timeframe <= now.
  - Nothing is filled, interpolated or spliced across exchanges.
"""
from __future__ import annotations

import time

import ccxt
import pandas as pd

from .symbols import QUOTE_PREFERENCE

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Transient -- worth retrying. Anything else (BadSymbol, ExchangeError...) raises at once.
RETRYABLE = (ccxt.RateLimitExceeded, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.NetworkError)
MAX_RETRIES = 5


def timeframe_ms(timeframe: str) -> int:
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def now_ms() -> int:
    return int(time.time() * 1000)


class CCXTProvider:
    name = "ccxt"

    def __init__(self, exchange: str = "coinbase"):
        self.exchange_id = exchange
        self.ex = getattr(ccxt, exchange)({"enableRateLimit": True})
        self._markets = None

    @property
    def markets(self) -> dict:
        if self._markets is None:
            self._markets = self.ex.load_markets()
        return self._markets

    @property
    def page_limit(self) -> int:
        """Max candles per request as CCXT reports it (Coinbase: 300)."""
        return (self.ex.features.get("spot", {}).get("fetchOHLCV") or {}).get("limit") or 300

    def pick_quote(self, base: str) -> str:
        for q in QUOTE_PREFERENCE:
            if f"{base}/{q}" in self.markets:
                return q
        raise ValueError(f"{base}: no {'/'.join(QUOTE_PREFERENCE)} market on {self.exchange_id}")

    def _fetch_page(self, pair: str, timeframe: str, since: int, limit: int) -> list[list]:
        for attempt in range(MAX_RETRIES):
            try:
                return self.ex.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
            except RETRYABLE as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                print(f"[market-data]   retry {attempt + 1}/{MAX_RETRIES - 1} in {wait}s ({type(e).__name__})")
                time.sleep(wait)
        raise AssertionError("unreachable")

    def fetch(self, pair: str, timeframe: str, start_ms: int, end_ms: int | None = None,
              include_incomplete: bool = False) -> pd.DataFrame:
        """Every bar with start_ms <= open_time < end_ms, paged forward through history.

        Loop:
          request `limit` bars from `since`
          keep bars inside [since, end)
          page empty     -> jump the window forward by limit bars. Coinbase serves a fixed
                            time window per request, so an empty page means "nothing in this
                            window" (before listing, or an outage), not "end of history".
                            The jump is bounded by end_ms, so this always terminates.
          page non-empty -> next since = newest open time + one bar
          exchange returns only bars older than `since`, or the newest open time does
          not advance -> raise. Never loop, never silently skip history.
        """
        if pair not in self.markets:
            raise ValueError(f"{pair} is not listed on {self.exchange_id}")

        step = timeframe_ms(timeframe)
        limit = self.page_limit
        now = now_ms()
        end_ms = min(end_ms or now, now + step)
        since = start_ms
        rows: list[list] = []

        while since < end_ms:
            raw = self._fetch_page(pair, timeframe, since, limit)
            page = [c for c in raw if since <= c[0] < end_ms]
            stalled = raw and raw[-1][0] < since
            if page and rows and page[-1][0] <= rows[-1][0]:
                stalled = True
            if stalled:
                raise RuntimeError(f"{pair} {timeframe}: pagination stalled at "
                                   f"{pd.Timestamp(since, unit='ms', tz='UTC')} (exchange ignored `since`)")
            if not page:
                since += limit * step
                continue
            newest = page[-1][0]
            rows.extend(page)
            since = newest + step

        df = pd.DataFrame(rows, columns=COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)

        if not include_incomplete:
            closed = df["timestamp"] + pd.Timedelta(milliseconds=step) <= pd.Timestamp(now, unit="ms", tz="UTC")
            df = df[closed].reset_index(drop=True)
        return df
