"""
Symbol resolution -- decides whether a request is an equity or a crypto pair.

Rules, in order (the first that matches wins):
  1. "EXCHANGE:BASEQUOTE"  TradingView-style, e.g. "COINBASE:BTCUSD"  -> crypto pair
  2. "BASE/QUOTE"          explicit pair, e.g. "BTC/USD"              -> crypto pair
  3. "BASE-QUOTE"          dash pair where BASE is a known crypto     -> crypto pair
  4. "BASE"                shorthand where BASE is a known crypto     -> crypto, quote picked
                           later by the provider from QUOTE_PREFERENCE
  5. anything else         equity / ETF / index, passed to yfinance as-is ("^GSPC")

Explicit pair syntax always overrides inference. USDT is never chosen by shorthand;
ask for "BTC/USDT" explicitly if you really want it.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_EXCHANGE = "coinbase"

# Shorthand resolves to the first quote the exchange lists. On Coinbase the USD book is
# canonical and BTC-USDC is served from the same book (verified: identical candles back to
# 2016), so USD is first -- it is the longest, least ambiguous series.
QUOTE_PREFERENCE = ("USD", "USDC")

# Extend freely. Only used to tell "SOL" (crypto) from "SO" (Southern Co.) style tickers.
CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "UNI", "AAVE",
    "LTC", "BCH", "DOT", "ATOM", "NEAR", "APT", "ARB", "OP", "SUI", "HBAR",
    "XLM", "FIL", "ETC", "SHIB", "PEPE", "HYPE", "USDC", "USDT",
}

KNOWN_QUOTES = ("USDC", "USDT", "USD", "EUR", "GBP", "BTC", "ETH")


@dataclass(frozen=True)
class Resolved:
    asset_type: str            # "equity" or "crypto"
    ticker: str                # "AAPL", "^GSPC", "BTC"
    quote: str | None = None   # "USD" for crypto once known, None for equities / shorthand
    exchange: str | None = None

    @property
    def pair(self) -> str | None:
        return f"{self.ticker}/{self.quote}" if self.quote else None


def _split_concatenated(s: str) -> tuple[str, str]:
    """'BTCUSD' -> ('BTC', 'USD'). Longest known quote suffix wins, so USDC beats USD."""
    for q in sorted(KNOWN_QUOTES, key=len, reverse=True):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)], q
    raise ValueError(f"cannot split {s!r} into base/quote")


def resolve(symbol: str, exchange: str | None = None) -> Resolved:
    s = symbol.strip().upper()

    if ":" in s:
        venue, rest = s.split(":", 1)
        base, quote = rest.split("/") if "/" in rest else _split_concatenated(rest)
        return Resolved("crypto", base, quote, exchange or venue.lower())

    if "/" in s:
        base, quote = s.split("/", 1)
        return Resolved("crypto", base, quote, exchange or DEFAULT_EXCHANGE)

    if "-" in s:
        base, quote = s.split("-", 1)
        if base in CRYPTO_BASES:
            return Resolved("crypto", base, quote, exchange or DEFAULT_EXCHANGE)

    if s in CRYPTO_BASES:
        return Resolved("crypto", s, None, exchange or DEFAULT_EXCHANGE)

    return Resolved("equity", s)
