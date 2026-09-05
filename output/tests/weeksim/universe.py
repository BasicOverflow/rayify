"""Tiny deterministic multi-source universe for weeksim."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WeekUniverse:
    equity_symbols: tuple[str, ...] = ("AAPL", "MSFT")
    etf_symbols: tuple[str, ...] = ("SPY",)
    fred_series: tuple[str, ...] = ("DFF", "DGS10", "UNRATE")
    align_symbols: tuple[str, ...] = ("AAPL",)

    @property
    def all_equity_etf(self) -> tuple[str, ...]:
        return self.equity_symbols + self.etf_symbols

    def apply_env(self) -> None:
        os.environ["MARKETS_TEST_PROFILE"] = "1"
        os.environ["MARKETS_TEST_EOD_FULL_PATH"] = "1"
        os.environ["MARKETS_TEST_SYMBOLS"] = ",".join(self.all_equity_etf)
        os.environ["MARKETS_TEST_ALIGN_SYMBOLS"] = ",".join(self.align_symbols)
        os.environ["MARKETS_TEST_FRED_SERIES"] = ",".join(self.fred_series)
        os.environ["MARKETS_TEST_YF_LIMIT"] = str(len(self.all_equity_etf))
        os.environ["MARKETS_TEST_EOD_YF_LIMIT"] = str(len(self.all_equity_etf))
        os.environ["MARKETS_TEST_QUALITY_LIMIT"] = str(
            len(self.all_equity_etf) + len(self.fred_series)
        )


DEFAULT_UNIVERSE = WeekUniverse()
