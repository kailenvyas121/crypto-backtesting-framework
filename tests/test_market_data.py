"""
Offline tests for the market-data layer. No network: a fake exchange stands in for CCXT.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crypto_data import CCXTProvider  # noqa: E402
from src.data_validation import validate  # noqa: E402
from src.market_data import MarketData  # noqa: E402
from src.symbols import resolve  # noqa: E402

DAY = 86_400_000
T0 = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)   # fake history stays in the past


class FakeExchange:
    """Coinbase-like: serves a fixed window of `limit` bars from `since`, 300 max."""
    features = {"spot": {"fetchOHLCV": {"limit": 300}}}

    def __init__(self, first_ms, n_bars, holes=(), stall=False):
        self.bars = [[first_ms + i * DAY, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0]
                     for i in range(n_bars) if i not in holes]
        self.stall = stall
        self.calls = 0

    def load_markets(self):
        return {"BTC/USD": {}, "BTC/USDC": {}}

    def fetch_ohlcv(self, pair, timeframe, since, limit):
        self.calls += 1
        if self.stall:
            return self.bars[:limit]
        return [b for b in self.bars if since <= b[0] < since + limit * DAY]


def provider(ex) -> CCXTProvider:
    p = CCXTProvider.__new__(CCXTProvider)
    p.exchange_id, p.ex, p._markets = "fake", ex, None
    return p


# ---------- symbols ----------

@pytest.mark.parametrize("sym, kind, ticker, quote, venue", [
    ("AAPL", "equity", "AAPL", None, None),
    ("^GSPC", "equity", "^GSPC", None, None),
    ("BTC", "crypto", "BTC", None, "coinbase"),
    ("eth/usd", "crypto", "ETH", "USD", "coinbase"),
    ("SOL-USDC", "crypto", "SOL", "USDC", "coinbase"),
    ("COINBASE:BTCUSDC", "crypto", "BTC", "USDC", "coinbase"),
    ("KRAKEN:ETHUSD", "crypto", "ETH", "USD", "kraken"),
])
def test_resolve(sym, kind, ticker, quote, venue):
    r = resolve(sym)
    assert (r.asset_type, r.ticker, r.quote, r.exchange) == (kind, ticker, quote, venue)


def test_shorthand_prefers_usd_never_usdt():
    assert provider(FakeExchange(T0, 1)).pick_quote("BTC") == "USD"


# ---------- pagination ----------

def test_pagination_walks_full_history_without_duplicates():
    ex = FakeExchange(T0, 1000)
    df = provider(ex).fetch("BTC/USD", "1d", T0, T0 + 1000 * DAY)
    assert len(df) == 1000 and df.timestamp.is_unique and df.timestamp.is_monotonic_increasing
    assert ex.calls == 4                     # ceil(1000 / 300)


def test_pagination_jumps_empty_windows_before_listing_and_in_outages():
    first = T0 + 400 * DAY                   # listed 400 days after the requested start
    holes = set(range(100, 450))             # a 350-day outage, longer than one page
    df = provider(FakeExchange(first, 800, holes)).fetch("BTC/USD", "1d", T0, first + 800 * DAY)
    assert len(df) == 800 - len(holes)
    assert df.timestamp.iloc[0] == pd.Timestamp(first, unit="ms", tz="UTC")


def test_pagination_raises_when_it_stops_advancing():
    with pytest.raises(RuntimeError, match="stalled"):
        provider(FakeExchange(T0, 1000, stall=True)).fetch("BTC/USD", "1d", T0, T0 + 1000 * DAY)


def test_incomplete_bar_excluded_by_default():
    today = pd.Timestamp.now(tz="UTC").normalize()
    first = int((today - pd.Timedelta(days=4)).timestamp() * 1000)
    p = provider(FakeExchange(first, 5))     # last bar is today's, still forming
    assert p.fetch("BTC/USD", "1d", first).timestamp.iloc[-1] == today - pd.Timedelta(days=1)
    assert p.fetch("BTC/USD", "1d", first, include_incomplete=True).timestamp.iloc[-1] == today


# ---------- validation ----------

def bars(n=10, start="2024-01-01"):
    ts = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0})


def test_clean_data_passes():
    assert validate(bars(), "1d", "crypto").status == "PASS"


@pytest.mark.parametrize("breaker, message", [
    (lambda d: pd.concat([d, d.iloc[[3]]]).sort_values("timestamp"), "duplicate timestamps"),
    (lambda d: d.iloc[::-1], "not sorted"),
    (lambda d: d.assign(close=d.close.where(d.index != 2)), "null values in close"),
    (lambda d: d.assign(low=d.low.where(d.index != 2, -1.0)), "non-positive price"),
    (lambda d: d.assign(high=d.high.where(d.index != 2, 50.0)), "high < low"),
    (lambda d: d.assign(close=d.close.where(d.index != 2, 150.0)), "close outside"),
])
def test_validation_catches_malformed_data(breaker, message):
    r = validate(breaker(bars()).reset_index(drop=True), "1d", "crypto")
    assert not r.passed and any(message in e for e in r.errors), r.errors


def test_crypto_gap_is_a_warning():
    r = validate(bars().drop(index=4).reset_index(drop=True), "1d", "crypto")
    assert r.passed and len(r.missing) == 1


def test_equity_weekends_and_holidays_are_not_gaps():
    # 2024-07-01 .. 2024-07-08: 4 July is an NYSE holiday, 6-7 July a weekend
    sessions = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08"]).tz_localize("UTC")
    df = pd.DataFrame({"timestamp": sessions, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    assert validate(df, "1d", "equity").status == "PASS"
    assert len(validate(df.drop(index=3).reset_index(drop=True), "1d", "equity").missing) == 1


# ---------- cache + incremental update ----------

def test_cache_first_download_then_incremental_update(tmp_path):
    ex = FakeExchange(T0, 10)
    data = MarketData(cache_dir=tmp_path)
    data._crypto["coinbase"] = provider(ex)

    first = data.get("BTC", start="2018-01-01")
    assert len(first) == 10 and (tmp_path / "crypto/coinbase/BTC_USD_1d.parquet").exists()
    assert data.info("BTC")["pair"] == "BTC/USD" and data.info("BTC")["exchange"] == "coinbase"

    ex.bars += [[T0 + i * DAY, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0] for i in (10, 11)]
    ex.bars[9][4] = 109.0                    # provider revises the last stored bar's close
    result = data.update("BTC")
    assert result["rows_added"] == 2
    assert result["revised_bars"] == [str(pd.Timestamp(T0 + 9 * DAY, unit="ms", tz="UTC"))]

    again = data.load("BTC")
    assert len(again) == 12 and again.timestamp.is_unique
    assert data.update("BTC")["rows_added"] == 0    # nothing new -> nothing appended


def test_update_mixing_utc_flavours_keeps_datetime_dtype(tmp_path):
    # regression: yfinance + parquet hand back different UTC tz objects; a naive concat -> object dtype
    import datetime as dt
    data = MarketData(cache_dir=tmp_path)
    data._crypto["coinbase"] = provider(FakeExchange(T0, 10))
    data.get("BTC", start="2018-01-01")
    real_fetch = data._fetch
    def fetch_with_stdlib_utc(*a, **k):
        df = real_fetch(*a, **k)
        return df.assign(timestamp=df.timestamp.dt.tz_convert(dt.timezone.utc))
    data._fetch = fetch_with_stdlib_utc
    assert data.update("BTC")["validation"] == "PASS"


def test_failing_dataset_is_never_written(tmp_path):
    ex = FakeExchange(T0, 10)
    data = MarketData(cache_dir=tmp_path)
    data._crypto["coinbase"] = provider(ex)
    data.get("BTC", start="2018-01-01")
    ex.bars.append([T0 + 10 * DAY, 100.0, 90.0, 110.0, 100.0, 1.0])   # high < low
    with pytest.raises(ValueError, match="not saved"):
        data.update("BTC")
    assert len(data.load("BTC")) == 10                                  # last good file intact
