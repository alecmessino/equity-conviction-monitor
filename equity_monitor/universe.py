"""Universe definition: tickers, names, GICS sectors, benchmark weights.

Primary source is ``data/universe.csv`` — an index-constituent export (an iShares
IWB/Russell 1000 holdings file works as-is, preamble rows and all). That file is
the only practical way to get a real GICS taxonomy for a thousand names; SEC's
SIC codes are coarser and older, and vendor sector strings arrive only for
whatever slice of the universe a rate limit happens to allow.

Without the file we fall back to a committed large-cap list so the repository
always builds and the tests always run.

ETFs are carried as ``asset_class="ETF"`` and deliberately excluded from
conviction scoring: an index fund has no ROIC, and the previous terminal
assigning SPY a "quality" score was a category error, not a data gap. They stay
in the universe as benchmarks and sector proxies.
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

BENCHMARK = "SPY"

# GICS sector -> liquid sector-proxy ETF, for sector-relative strength.
SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Consumer Staples": "XLP",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Vendors spell GICS sectors inconsistently; normalise to the canonical labels above.
_SECTOR_ALIASES = {
    "technology": "Information Technology",
    "information technology": "Information Technology",
    "info tech": "Information Technology",
    "financial services": "Financials",
    "financials": "Financials",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "energy": "Energy",
    "consumer staples": "Consumer Staples",
    "consumer defensive": "Consumer Staples",
    "consumer discretionary": "Consumer Discretionary",
    "consumer cyclical": "Consumer Discretionary",
    "industrials": "Industrials",
    "utilities": "Utilities",
    "materials": "Materials",
    "basic materials": "Materials",
    "real estate": "Real Estate",
    "communication": "Communication Services",
    "communication services": "Communication Services",
}


@dataclass
class Member:
    symbol: str
    name: str = ""
    sector: str = ""
    asset_class: str = "Equity"
    index_weight: float | None = None
    vendor_pe: float | None = None

    @property
    def is_etf(self) -> bool:
        return self.asset_class.upper() == "ETF"

    def to_dict(self) -> dict:
        return asdict(self)


def normalise_sector(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    return _SECTOR_ALIASES.get(s.lower(), s.title())


def _clean_number(raw) -> float | None:
    if raw is None:
        return None
    s = str(raw).replace(",", "").replace("%", "").replace("$", "").strip()
    if not s or s in {"-", "--", "N/A", "n/a"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _find_header(rows: list[list[str]]) -> int:
    """Locate the real header row.

    iShares exports open with several lines of fund metadata before the table, so
    we scan for the first row that names a ticker column.
    """
    for i, row in enumerate(rows[:40]):
        joined = ",".join(c.lower() for c in row)
        if re.search(r"\b(ticker|symbol)\b", joined):
            return i
    return 0


def _column(header: list[str], *candidates: str) -> int | None:
    lowered = [h.strip().lower() for h in header]
    for cand in candidates:
        for i, h in enumerate(lowered):
            if h == cand:
                return i
    for cand in candidates:  # substring match as a second pass
        for i, h in enumerate(lowered):
            if cand in h:
                return i
    return None


def parse_constituents(text: str) -> list[Member]:
    """Parse an index-holdings CSV into Members. Tolerant of vendor layouts."""
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        return []
    h = _find_header(rows)
    header = rows[h]
    idx_sym = _column(header, "ticker", "symbol")
    if idx_sym is None:
        return []
    idx_name = _column(header, "name", "security", "description")
    idx_sector = _column(header, "sector", "gics sector")
    idx_class = _column(header, "asset class", "asset_class", "type")
    idx_weight = _column(header, "weight (%)", "weight", "% of net assets")
    idx_pe = _column(header, "p/e", "pe", "price/earnings")

    def cell(row: list[str], i: int | None) -> str:
        return row[i].strip() if (i is not None and i < len(row)) else ""

    seen: set[str] = set()
    out: list[Member] = []
    for row in rows[h + 1:]:
        sym = cell(row, idx_sym).upper().replace(" ", "")
        # Index files carry cash and futures line items with placeholder tickers.
        if not sym or sym in seen or not re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", sym):
            continue
        asset_class = cell(row, idx_class) or "Equity"
        sector_raw = cell(row, idx_sector)
        # Allowlist, not denylist. A holdings file carries cash, money-market,
        # futures and collateral lines under labels that vary by vendor and change
        # without notice ("Cash Collateral and Margins" is one this file uses), so
        # enumerating what to reject leaks a new one in every time.
        if asset_class.strip().lower() not in {"equity", "etf", "fund", "stock"}:
            continue
        if "derivative" in sector_raw.lower() or "cash" in sector_raw.lower():
            continue
        seen.add(sym)
        out.append(Member(
            symbol=sym,
            name=cell(row, idx_name),
            sector=normalise_sector(cell(row, idx_sector)),
            asset_class="ETF" if "etf" in asset_class.lower() else "Equity",
            index_weight=_clean_number(cell(row, idx_weight)),
            vendor_pe=_clean_number(cell(row, idx_pe)),
        ))
    return out


def _constituents_file() -> str | None:
    """First readable constituents file in data/, if any."""
    if not os.path.isdir(DATA_DIR):
        return None
    for name in sorted(os.listdir(DATA_DIR)):
        if name.lower().endswith(".csv") and not name.startswith("."):
            return os.path.join(DATA_DIR, name)
    return None


# ---------------------------------------------------------------------------
# fallback universe (used when data/ has no constituents file)
# ---------------------------------------------------------------------------
_T, _F, _H, _E, _P, _D, _I, _U, _M, _R, _C = (
    "Information Technology", "Financials", "Health Care", "Energy",
    "Consumer Staples", "Consumer Discretionary", "Industrials", "Utilities",
    "Materials", "Real Estate", "Communication Services",
)

FALLBACK: list[tuple[str, str, str]] = [
    ("AAPL", "Apple Inc.", _T), ("MSFT", "Microsoft Corp.", _T),
    ("NVDA", "NVIDIA Corp.", _T), ("AVGO", "Broadcom Inc.", _T),
    ("ORCL", "Oracle Corp.", _T), ("CRM", "Salesforce Inc.", _T),
    ("AMD", "Advanced Micro Devices", _T), ("ADBE", "Adobe Inc.", _T),
    ("CSCO", "Cisco Systems", _T), ("ACN", "Accenture plc", _T),
    ("TXN", "Texas Instruments", _T), ("QCOM", "Qualcomm Inc.", _T),
    ("INTC", "Intel Corp.", _T), ("IBM", "IBM Corp.", _T),
    ("AMAT", "Applied Materials", _T), ("MU", "Micron Technology", _T),
    ("NOW", "ServiceNow Inc.", _T), ("INTU", "Intuit Inc.", _T),
    ("LRCX", "Lam Research", _T), ("KLAC", "KLA Corp.", _T),
    ("GOOGL", "Alphabet Inc. Class A", _C), ("META", "Meta Platforms", _C),
    ("NFLX", "Netflix Inc.", _C), ("DIS", "Walt Disney Co.", _C),
    ("CMCSA", "Comcast Corp.", _C), ("T", "AT&T Inc.", _C),
    ("VZ", "Verizon Communications", _C), ("TMUS", "T-Mobile US", _C),
    ("AMZN", "Amazon.com Inc.", _D), ("TSLA", "Tesla Inc.", _D),
    ("HD", "Home Depot Inc.", _D), ("MCD", "McDonald's Corp.", _D),
    ("NKE", "Nike Inc.", _D), ("LOW", "Lowe's Companies", _D),
    ("SBUX", "Starbucks Corp.", _D), ("BKNG", "Booking Holdings", _D),
    ("TJX", "TJX Companies", _D), ("GM", "General Motors", _D),
    ("F", "Ford Motor Co.", _D),
    ("BRK.B", "Berkshire Hathaway B", _F), ("JPM", "JPMorgan Chase", _F),
    ("V", "Visa Inc.", _F), ("MA", "Mastercard Inc.", _F),
    ("BAC", "Bank of America", _F), ("WFC", "Wells Fargo & Co.", _F),
    ("GS", "Goldman Sachs Group", _F), ("MS", "Morgan Stanley", _F),
    ("SPGI", "S&P Global Inc.", _F), ("BLK", "BlackRock Inc.", _F),
    ("AXP", "American Express", _F), ("C", "Citigroup Inc.", _F),
    ("SCHW", "Charles Schwab", _F), ("CB", "Chubb Ltd.", _F),
    ("PGR", "Progressive Corp.", _F),
    ("LLY", "Eli Lilly & Co.", _H), ("UNH", "UnitedHealth Group", _H),
    ("JNJ", "Johnson & Johnson", _H), ("ABBV", "AbbVie Inc.", _H),
    ("MRK", "Merck & Co.", _H), ("TMO", "Thermo Fisher Scientific", _H),
    ("ABT", "Abbott Laboratories", _H), ("PFE", "Pfizer Inc.", _H),
    ("AMGN", "Amgen Inc.", _H), ("DHR", "Danaher Corp.", _H),
    ("ISRG", "Intuitive Surgical", _H), ("BMY", "Bristol-Myers Squibb", _H),
    ("GILD", "Gilead Sciences", _H), ("CVS", "CVS Health Corp.", _H),
    ("MDT", "Medtronic plc", _H), ("ELV", "Elevance Health", _H),
    ("XOM", "Exxon Mobil Corp.", _E), ("CVX", "Chevron Corp.", _E),
    ("COP", "ConocoPhillips", _E), ("SLB", "Schlumberger NV", _E),
    ("EOG", "EOG Resources", _E), ("PSX", "Phillips 66", _E),
    ("WMT", "Walmart Inc.", _P), ("PG", "Procter & Gamble", _P),
    ("COST", "Costco Wholesale", _P), ("KO", "Coca-Cola Co.", _P),
    ("PEP", "PepsiCo Inc.", _P), ("PM", "Philip Morris Intl", _P),
    ("MDLZ", "Mondelez International", _P), ("MO", "Altria Group", _P),
    ("CL", "Colgate-Palmolive", _P), ("KMB", "Kimberly-Clark", _P),
    ("GE", "GE Aerospace", _I), ("CAT", "Caterpillar Inc.", _I),
    ("RTX", "RTX Corp.", _I), ("HON", "Honeywell International", _I),
    ("UNP", "Union Pacific Corp.", _I), ("BA", "Boeing Co.", _I),
    ("LMT", "Lockheed Martin", _I), ("DE", "Deere & Co.", _I),
    ("UPS", "United Parcel Service", _I), ("ETN", "Eaton Corp. plc", _I),
    ("MMM", "3M Co.", _I), ("ADP", "Automatic Data Processing", _I),
    ("NEE", "NextEra Energy", _U), ("DUK", "Duke Energy Corp.", _U),
    ("SO", "Southern Co.", _U), ("D", "Dominion Energy", _U),
    ("LIN", "Linde plc", _M), ("SHW", "Sherwin-Williams", _M),
    ("APD", "Air Products & Chemicals", _M), ("FCX", "Freeport-McMoRan", _M),
    ("NEM", "Newmont Corp.", _M),
    ("PLD", "Prologis Inc.", _R), ("AMT", "American Tower", _R),
    ("EQIX", "Equinix Inc.", _R), ("SPG", "Simon Property Group", _R),
]

# Benchmarks, factor sleeves, and sector proxies. Never conviction-scored.
ETFS: list[tuple[str, str]] = [
    ("SPY", "S&P 500 ETF"), ("QQQ", "Nasdaq 100 ETF"), ("IWM", "Russell 2000 ETF"),
    ("IWB", "Russell 1000 ETF"), ("IWD", "Russell 1000 Value ETF"),
    ("EFA", "MSCI EAFE ETF"), ("EEM", "MSCI Emerging Markets ETF"),
    ("VWO", "FTSE Emerging Markets ETF"), ("EWJ", "MSCI Japan ETF"),
    ("QUAL", "MSCI USA Quality Factor ETF"), ("VLUE", "MSCI USA Value Factor ETF"),
    ("MTUM", "MSCI USA Momentum Factor ETF"), ("USMV", "MSCI USA Min Vol ETF"),
    ("XLK", "Technology Select Sector"), ("XLF", "Financial Select Sector"),
    ("XLV", "Health Care Select Sector"), ("XLE", "Energy Select Sector"),
    ("XLP", "Consumer Staples Select Sector"), ("XLY", "Consumer Discretionary Select Sector"),
    ("XLI", "Industrial Select Sector"), ("XLU", "Utilities Select Sector"),
    ("XLB", "Materials Select Sector"), ("XLRE", "Real Estate Select Sector"),
    ("XLC", "Communication Services Select Sector"),
]


def load(limit: int | None = None) -> tuple[list[Member], str]:
    """Return (members, provenance).

    `limit` caps the number of *equities* (ETFs are always included — they are
    benchmarks, not candidates, and dropping them would break sector RS).
    """
    path = _constituents_file()
    members: list[Member] = []
    provenance = ""
    if path:
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as fh:
                members = parse_constituents(fh.read())
            provenance = f"data/{os.path.basename(path)}"
        except Exception:
            members = []
    if not members:
        members = [Member(symbol=s, name=n, sector=sec) for s, n, sec in FALLBACK]
        provenance = "built-in large-cap fallback"

    equities = [m for m in members if not m.is_etf]
    equities.sort(key=lambda m: (m.index_weight is None, -(m.index_weight or 0.0), m.symbol))
    if limit:
        equities = equities[:limit]

    have = {m.symbol for m in equities}
    etfs = [Member(symbol=s, name=n, asset_class="ETF") for s, n in ETFS if s not in have]
    # A sector proxy the constituents file happened to include stays an ETF.
    etfs += [m for m in members if m.is_etf and m.symbol not in have
             and m.symbol not in {e.symbol for e in etfs}]
    return equities + etfs, provenance


def sector_proxy(sector: str) -> str:
    return SECTOR_ETF.get(sector, BENCHMARK)
