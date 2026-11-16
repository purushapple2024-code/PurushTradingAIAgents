"""Index constituent lists.

Static snapshots are shipped so the system runs offline; `refresh_from_nse()` updates them
from NSE's public index pages when network is available. Snapshot date: November 2026.
Constituents change a few times a year, so refresh before training.
"""

from __future__ import annotations

import json
from pathlib import Path

NIFTY50: list[str] = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SHRIRAMFIN",
    "SBIN",
    "SUNPHARMA",
    "TCS",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
]

# NIFTY Next 50 snapshot; NIFTY100 = NIFTY50 + NIFTYNEXT50.
NIFTYNEXT50: list[str] = [
    "ABB",
    "ADANIENSOL",
    "ADANIGREEN",
    "ADANIPOWER",
    "AMBUJACEM",
    "BAJAJHLDNG",
    "BANKBARODA",
    "BPCL",
    "BRITANNIA",
    "CANBK",
    "CGPOWER",
    "CHOLAFIN",
    "DABUR",
    "DIVISLAB",
    "DLF",
    "DMART",
    "GAIL",
    "GODREJCP",
    "HAVELLS",
    "HAL",
    "HYUNDAI",
    "ICICIGI",
    "ICICIPRULI",
    "INDHOTEL",
    "INDIGO",
    "IOC",
    "IRFC",
    "JINDALSTEL",
    "JSWENERGY",
    "LICI",
    "LODHA",
    "LTIM",
    "MOTHERSON",
    "NAUKRI",
    "PFC",
    "PIDILITIND",
    "PNB",
    "RECLTD",
    "SHREECEM",
    "SIEMENS",
    "SWIGGY",
    "TATAPOWER",
    "TORNTPHARM",
    "TVSMOTOR",
    "UNITDSPR",
    "VBL",
    "VEDL",
    "ZYDUSLIFE",
    "BOSCHLTD",
    "UNIONBANK",
]

NIFTY100: list[str] = NIFTY50 + NIFTYNEXT50

# Most liquid NIFTY 50 names by typical daily turnover; intraday default. Configurable.
NIFTY50_LIQUID15: list[str] = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "TCS",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "BHARTIARTL",
    "LT",
    "TATAMOTORS",
    "BAJFINANCE",
    "ITC",
    "HINDUNILVR",
    "TATASTEEL",
]

# Sector map for the sector-exposure rule (NSE sector classification, simplified).
SECTOR: dict[str, str] = {
    **dict.fromkeys(
        ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK"], "Banks"
    ),
    **dict.fromkeys(
        ["BAJFINANCE", "BAJAJFINSV", "JIOFIN", "SHRIRAMFIN", "HDFCLIFE", "SBILIFE"],
        "Financial Services",
    ),
    **dict.fromkeys(["INFY", "TCS", "HCLTECH", "TECHM", "WIPRO"], "IT"),
    **dict.fromkeys(["RELIANCE", "ONGC", "COALINDIA"], "Oil Gas & Energy"),
    **dict.fromkeys(["NTPC", "POWERGRID"], "Power"),
    **dict.fromkeys(
        ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO"], "Automobile"
    ),
    **dict.fromkeys(["SUNPHARMA", "CIPLA", "DRREDDY", "APOLLOHOSP"], "Healthcare"),
    **dict.fromkeys(["HINDUNILVR", "ITC", "NESTLEIND", "TATACONSUM"], "FMCG"),
    **dict.fromkeys(["TATASTEEL", "JSWSTEEL", "HINDALCO"], "Metals"),
    **dict.fromkeys(["LT", "ADANIPORTS", "ADANIENT", "BEL"], "Infrastructure"),
    **dict.fromkeys(["ULTRACEMCO", "GRASIM", "ASIANPAINT"], "Materials"),
    **dict.fromkeys(["BHARTIARTL"], "Telecom"),
    **dict.fromkeys(["TITAN", "TRENT", "ETERNAL"], "Consumer"),
}

UNIVERSES: dict[str, list[str]] = {
    "NIFTY50": NIFTY50,
    "NIFTYNEXT50": NIFTYNEXT50,
    "NIFTY100": NIFTY100,
    "NIFTY50_LIQUID15": NIFTY50_LIQUID15,
}


def resolve_universe(name: str, custom: list[str] | None = None) -> list[str]:
    if name.lower() == "custom":
        if not custom:
            raise ValueError("universe 'custom' selected but config.universe.custom is empty")
        return [s.upper() for s in custom]
    try:
        return list(UNIVERSES[name.upper()])
    except KeyError as e:
        raise ValueError(
            f"unknown universe {name!r}; choose from {sorted(UNIVERSES)} or custom"
        ) from e


def sector_of(symbol: str) -> str:
    return SECTOR.get(symbol.upper(), "Other")


def to_yf(symbol: str) -> str:
    """NSE symbol → Yahoo Finance ticker (M&M → M&M.NS works; BAJAJ-AUTO → BAJAJ-AUTO.NS)."""
    return f"{symbol.upper()}.NS"


def save_snapshot(path: Path) -> None:
    path.write_text(json.dumps(UNIVERSES, indent=2), encoding="utf-8")
