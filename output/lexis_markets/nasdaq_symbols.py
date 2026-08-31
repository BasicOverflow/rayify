from __future__ import annotations

from dataclasses import dataclass, field

import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


@dataclass
class NasdaqEntry:
    symbol: str
    exchange: str
    etf: bool


@dataclass
class NasdaqDirectory:
    symbols: set[str] = field(default_factory=set)
    by_symbol: dict[str, NasdaqEntry] = field(default_factory=dict)

    def add(self, symbol: str, exchange: str, etf: bool) -> None:
        sym = symbol.strip().upper()
        if not sym:
            return
        self.symbols.add(sym)
        self.by_symbol[sym] = NasdaqEntry(symbol=sym, exchange=exchange.strip().upper(), etf=etf)


def _parse_nasdaq_listed(text: str) -> NasdaqDirectory:
    out = NasdaqDirectory()
    lines = text.splitlines()
    if not lines:
        return out
    header = lines[0].split("|")
    sym_i = header.index("Symbol")
    test_i = header.index("Test Issue")
    etf_i = header.index("ETF")
    for line in lines[1:]:
        if not line or line.startswith("File Creation Time"):
            break
        parts = line.split("|")
        if len(parts) <= max(sym_i, test_i, etf_i):
            continue
        if parts[test_i].strip().upper() != "N":
            continue
        out.add(parts[sym_i], "Q", parts[etf_i].strip().upper() == "Y")
    return out


def _parse_other_listed(text: str) -> NasdaqDirectory:
    out = NasdaqDirectory()
    lines = text.splitlines()
    if not lines:
        return out
    header = lines[0].split("|")
    sym_i = header.index("ACT Symbol")
    test_i = header.index("Test Issue")
    exch_i = header.index("Exchange")
    etf_i = header.index("ETF")
    for line in lines[1:]:
        if not line or line.startswith("File Creation Time"):
            break
        parts = line.split("|")
        if len(parts) <= max(sym_i, test_i, exch_i, etf_i):
            continue
        if parts[test_i].strip().upper() != "N":
            continue
        out.add(parts[sym_i], parts[exch_i], parts[etf_i].strip().upper() == "Y")
    return out


def fetch_nasdaq_directory() -> NasdaqDirectory:
    nasdaq = requests.get(NASDAQ_LISTED_URL, timeout=60)
    nasdaq.raise_for_status()
    other = requests.get(OTHER_LISTED_URL, timeout=60)
    other.raise_for_status()
    out = _parse_nasdaq_listed(nasdaq.text)
    for sym, entry in _parse_other_listed(other.text).by_symbol.items():
        out.add(sym, entry.exchange, entry.etf)
    return out


def fetch_nasdaq_listed_symbols() -> set[str]:
    return fetch_nasdaq_directory().symbols
