"""
MarketData -- the one entry point research code uses for prices.

    from src.market_data import MarketData
    data = MarketData()
    btc = data.get("BTC", start="2018-01-01")            # Coinbase BTC/USD, 1d
    spy = data.get("SPY", start="2010-01-01")            # yfinance, 1d
    eth = data.get("ETH/USD", start="2024-01-01", timeframe="1h")

Strategies never import yfinance or ccxt; they get a normalised frame from here.

Cache (data/raw/market/, API output only, never hand-edited):
    equities/SPY_1d.parquet                + SPY_1d.meta.json
    crypto/coinbase/BTC_USD_1d.parquet     + BTC_USD_1d.meta.json

get()    reads the cache; fetches only what is missing (head backfill, tail update).
load()   reads the cache only. Never touches the network -- builders use this.
update() refetches the last OVERLAP_BARS stored bars plus anything newer, reports any
         stored value the provider has since revised, then appends. Revisions deeper than
         the overlap are never rewritten silently.

Dates: `start` / `end` are inclusive and refer to bar timestamps (open times, UTC).
end="2026-08-31" on daily data includes the 31 August bar.

Only closed bars are ever written to the cache. include_incomplete=True appends the
currently forming bar to the returned frame without storing it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .crypto_data import CCXTProvider, timeframe_ms
from .data_validation import ValidationReport, validate
from .equity_data import YFinanceProvider
from .symbols import DEFAULT_EXCHANGE, QUOTE_PREFERENCE, Resolved, resolve

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "market"
OVERLAP_BARS = 3
EARLIEST = pd.Timestamp("2010-01-01", tz="UTC")   # default start when none is given


def _ts(x) -> pd.Timestamp | None:
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _utc(df: pd.DataFrame) -> pd.DataFrame:
    """One timestamp dtype everywhere. Providers and parquet hand back UTC as different tz
    objects (datetime.timezone.utc vs zoneinfo 'UTC'); concatenating the two silently turns
    the column into dtype=object."""
    return df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("UTC").dt.as_unit("ms"))


def _log(msg: str) -> None:
    print(f"[market-data] {msg}")


class MarketData:
    def __init__(self, exchange: str = DEFAULT_EXCHANGE, cache_dir: Path = CACHE):
        self.exchange = exchange
        self.cache_dir = Path(cache_dir)
        self._crypto: dict[str, CCXTProvider] = {}
        self._equity = YFinanceProvider()

    # ---------- resolution and paths ----------

    def _crypto_provider(self, exchange: str) -> CCXTProvider:
        if exchange not in self._crypto:
            self._crypto[exchange] = CCXTProvider(exchange)
        return self._crypto[exchange]

    def resolve(self, symbol: str, timeframe: str = "1d", offline: bool = False) -> Resolved:
        """Shorthand crypto ("BTC") needs a quote. Online: ask the exchange which of
        QUOTE_PREFERENCE it lists. Offline: use whichever of them is already cached."""
        r = resolve(symbol, exchange=None if ":" in symbol else self.exchange)
        if r.asset_type != "crypto" or r.quote is not None:
            return r
        if offline:
            for q in QUOTE_PREFERENCE:
                cand = Resolved("crypto", r.ticker, q, r.exchange)
                if self._paths(cand, timeframe)[0].exists():
                    return cand
            raise FileNotFoundError(f"{symbol} {timeframe} is not cached")
        return Resolved("crypto", r.ticker, self._crypto_provider(r.exchange).pick_quote(r.ticker), r.exchange)

    def _paths(self, r: Resolved, timeframe: str) -> tuple[Path, Path]:
        if r.asset_type == "crypto":
            stem = self.cache_dir / "crypto" / r.exchange / f"{r.ticker}_{r.quote}_{timeframe}"
        else:
            stem = self.cache_dir / "equities" / f"{r.ticker.replace('^', 'IDX_')}_{timeframe}"
        return stem.with_suffix(".parquet"), stem.with_suffix(".meta.json")

    def _read(self, r: Resolved, timeframe: str) -> tuple[pd.DataFrame | None, dict]:
        data, meta = self._paths(r, timeframe)
        if not data.exists():
            return None, {}
        return _utc(pd.read_parquet(data)), json.loads(meta.read_text())

    def _fetch(self, r: Resolved, timeframe: str, start: pd.Timestamp, end: pd.Timestamp | None = None,
               include_incomplete: bool = False) -> pd.DataFrame:
        if r.asset_type == "crypto":
            end_ms = int(end.timestamp() * 1000) if end is not None else None
            df = self._crypto_provider(r.exchange).fetch(
                r.pair, timeframe, int(start.timestamp() * 1000), end_ms, include_incomplete)
        else:
            df = self._equity.fetch(r.ticker, timeframe, start, end, include_incomplete)
        return _utc(df)

    def _write(self, r: Resolved, timeframe: str, df: pd.DataFrame, meta: dict) -> ValidationReport:
        """Validate, then save. A dataset that FAILS is never written: the previous good
        file stays on disk and the report is raised for investigation."""
        report = validate(df, timeframe, r.asset_type)
        if not report.passed:
            raise ValueError(f"{r.pair or r.ticker} {timeframe}: not saved, validation failed\n{report}")
        data, meta_path = self._paths(r, timeframe)
        data.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data, index=False)
        meta.update({
            "asset_type": r.asset_type,
            "provider": "ccxt" if r.asset_type == "crypto" else "yfinance",
            "exchange": r.exchange,
            "ticker": r.ticker,
            "pair": r.pair,
            "timeframe": timeframe,
            "timestamp_convention": "bar open time, UTC" if r.asset_type == "crypto"
                                    else "session date at 00:00 UTC (close = official XNYS close)"
                                    if timeframe == "1d" else "bar open time, UTC",
            "rows": len(df),
            "first_timestamp": str(df["timestamp"].iloc[0]),
            "last_timestamp": str(df["timestamp"].iloc[-1]),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "validation": report.status,
            "missing_bars": len(report.missing),
            "local_file": str(data.relative_to(ROOT)) if data.is_relative_to(ROOT) else str(data),
        })
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        return report

    # ---------- public API ----------

    def load(self, symbol: str, timeframe: str = "1d", start=None, end=None) -> pd.DataFrame:
        """Cache only. Raises if the dataset has never been downloaded."""
        df, _ = self._read(self.resolve(symbol, timeframe, offline=True), timeframe)
        if df is None:
            raise FileNotFoundError(f"{symbol} {timeframe} is not cached -- run data.get() or "
                                    f"scripts/update_market_data.py first")
        return self._slice(df, start, end)

    def get(self, symbol: str, start=None, end=None, timeframe: str = "1d",
            include_incomplete: bool = False) -> pd.DataFrame:
        r = self.resolve(symbol)
        start, end = _ts(start) or EARLIEST, _ts(end)
        df, meta = self._read(r, timeframe)
        label = f"{r.pair or r.ticker} {timeframe}"

        if df is None:
            provider = f"ccxt / {r.exchange}" if r.asset_type == "crypto" else "yfinance"
            _log(f"{label}  first download from {start.date()}  ({provider})")
            df = self._fetch(r, timeframe, start)
            if df.empty:
                raise ValueError(f"{label}: provider returned no data")
            report = self._write(r, timeframe, df, {"requested_start": str(start)})
            _log(f"{label}  rows: {len(df):,}  {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")
            _log(f"{label}  {report}")
        else:
            # head: only backfill if we have never asked this far back before
            if start < _ts(meta["requested_start"]):
                older = self._fetch(r, timeframe, start, df["timestamp"].iloc[0])
                df = pd.concat([older, df]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
                self._write(r, timeframe, df.reset_index(drop=True), {**meta, "requested_start": str(start)})
                _log(f"{label}  backfilled {len(older):,} older bars")
            # tail: only update if the request reaches past what is stored
            if (end is None or end > df["timestamp"].iloc[-1]) and self._newer_bar_possible(r, timeframe, df):
                self.update(symbol, timeframe, quiet=True)
                df, _ = self._read(r, timeframe)

        out = self._slice(df, start, end)
        if include_incomplete:
            live = self._fetch(r, timeframe, df["timestamp"].iloc[-1], include_incomplete=True)
            live = live[live["timestamp"] > df["timestamp"].iloc[-1]]
            out = pd.concat([out, live], ignore_index=True)
        return out

    def update(self, symbol: str, timeframe: str = "1d", quiet: bool = False) -> dict:
        r = self.resolve(symbol)
        df, meta = self._read(r, timeframe)
        label = f"{r.pair or r.ticker} {timeframe}"
        if df is None:
            raise FileNotFoundError(f"{label} is not cached -- call get() once to create it")

        old_last = df["timestamp"].iloc[-1]
        overlap_start = df["timestamp"].iloc[max(0, len(df) - OVERLAP_BARS)]
        fresh = self._fetch(r, timeframe, overlap_start)

        # compare the overlap: a changed stored bar is a revision, reported not hidden
        cols = [c for c in df.columns if c != "timestamp"]
        both = df.merge(fresh, on="timestamp", suffixes=("_old", "_new"))
        revised = [str(row.timestamp) for row in both.itertuples()
                   if any(abs(getattr(row, f"{c}_old") - getattr(row, f"{c}_new")) > 1e-9 for c in cols)]

        new = fresh[fresh["timestamp"] > old_last]
        merged = (pd.concat([df, fresh]).drop_duplicates("timestamp", keep="last")
                  .sort_values("timestamp").reset_index(drop=True))
        if revised:
            meta.setdefault("revisions", []).append(
                {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bars": revised})
        report = self._write(r, timeframe, merged, meta)

        result = {"asset": label, "provider": meta["provider"], "exchange": r.exchange,
                  "old_last": old_last, "new_last": merged["timestamp"].iloc[-1],
                  "rows_before": len(df), "rows_added": len(new), "revised_bars": revised,
                  "validation": report.status}
        if not quiet or len(new) or revised:
            _log(f"{label}")
            print(f"  provider: {meta['provider']}" + (f"   exchange: {r.exchange}" if r.exchange else ""))
            print(f"  existing rows: {len(df):,}   last stored: {old_last}")
            print(f"  new rows: {len(new)}   latest: {result['new_last']}")
            if revised:
                print(f"  REVISED by provider since last pull: {', '.join(revised)}")
            print(f"  {report}")
        return result

    def update_all(self) -> list[dict]:
        results = []
        for meta_path in sorted(self.cache_dir.rglob("*.meta.json")):
            m = json.loads(meta_path.read_text())
            symbol = f"{m['exchange'].upper()}:{m['pair']}" if m["asset_type"] == "crypto" else m["ticker"]
            results.append(self.update(symbol, m["timeframe"]))
        return results

    def validate(self, symbol: str, timeframe: str = "1d") -> ValidationReport:
        r = self.resolve(symbol, timeframe, offline=True)
        return validate(self.load(symbol, timeframe), timeframe, r.asset_type)

    def info(self, symbol: str, timeframe: str = "1d") -> dict:
        _, meta = self._read(self.resolve(symbol, timeframe, offline=True), timeframe)
        return meta

    @staticmethod
    def _newer_bar_possible(r: Resolved, timeframe: str, df: pd.DataFrame) -> bool:
        """Crypto: skip the network if the bar after the last stored one has not closed yet.
        Equities: always check -- sessions, holidays and Yahoo delays make it unpredictable."""
        if r.asset_type != "crypto":
            return True
        step = pd.Timedelta(milliseconds=timeframe_ms(timeframe))
        return df["timestamp"].iloc[-1] + 2 * step <= pd.Timestamp.now(tz="UTC")

    @staticmethod
    def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
        start, end = _ts(start), _ts(end)
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= df["timestamp"] >= start
        if end is not None:
            # a bare date means "through that day", so an intraday series keeps the whole day
            day_end = end + pd.Timedelta(days=1) if end == end.normalize() else end + pd.Timedelta(microseconds=1)
            mask &= df["timestamp"] < day_end
        return df[mask].reset_index(drop=True)
