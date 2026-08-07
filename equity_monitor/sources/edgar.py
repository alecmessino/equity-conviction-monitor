"""SEC EDGAR XBRL fundamentals — keyless, quota-free, filing-grade.

Why the ``frames`` API rather than ``companyfacts``
---------------------------------------------------
``companyfacts/CIK##########.json`` is ~4 MB per company; covering a 1,000-name
universe would mean ~4 GB per run. ``frames`` inverts the axis: one request
returns *every* filer's value for a single concept in a single period, gzipped
to ~60 KB. Twelve concepts across a handful of periods covers the entire
universe in well under a hundred requests. That is the difference between a
pipeline that scales to the Russell 1000 and one that does not.

Every number here traces to an XBRL fact in a real 10-K/10-Q, and each carries
the accession number and period end date it came from so the terminal can cite
its source rather than asking the reader to trust it.

Tag fallbacks matter: filers tag the same economic concept differently
(``Revenues`` vs ``RevenueFromContractWithCustomerExcludingAssessedTax``), so
each metric declares an ordered list of acceptable tags and we take the first
that resolves for a given company.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

from ._http import cached_json, sec_user_agent

FRAMES = "https://data.sec.gov/api/xbrl/frames/{tax}/{concept}/{unit}/{period}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _sec_kw() -> dict:
    """Request kwargs for SEC endpoints. SEC asks for fewer than 10 requests/second."""
    return dict(headers={"User-Agent": sec_user_agent()},
                throttle_key="sec", min_interval=0.12)

# US federal statutory corporate rate, used to convert EBIT to NOPAT. A single
# blended rate across the universe keeps ROIC cross-sectionally comparable, which
# is what the ranking needs; per-company effective rates would add noise from
# one-off tax items without improving the ordering.
TAX_RATE = 0.21

# ---------------------------------------------------------------------------
# concept definitions: (metric, taxonomy, unit, [tag fallbacks in priority order])
# ---------------------------------------------------------------------------
FLOW_CONCEPTS: list[tuple[str, str, str, list[str]]] = [
    ("revenue", "us-gaap", "USD", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ]),
    ("gross_profit", "us-gaap", "USD", ["GrossProfit"]),
    # Many filers report cost of revenue but never tag GrossProfit; we reconstruct it.
    ("cost_of_revenue", "us-gaap", "USD", [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ]),
    ("operating_income", "us-gaap", "USD", ["OperatingIncomeLoss"]),
    ("net_income", "us-gaap", "USD", [
        "NetIncomeLoss",
        "ProfitLoss",
    ]),
    # Income attributable to common holders. Better covered than NetIncomeLoss for
    # banks (11/11 vs 9/11 of the large US banks) and for REITs, where preferred
    # distributions make the distinction material.
    ("net_income_common", "us-gaap", "USD", [
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "NetIncomeLoss",
    ]),
    # --- banking income statement ---
    # Banks do not report OperatingIncomeLoss or GrossProfit at all (0/11 of the
    # large US banks tag either), which is why the default quality inputs are blank
    # for the whole sector. These three are tagged by 10-11 of 11.
    ("net_interest_income", "us-gaap", "USD", [
        "InterestIncomeExpenseNet",
        "InterestIncomeExpenseAfterProvisionForLoanLoss",
    ]),
    ("noninterest_expense", "us-gaap", "USD", ["NoninterestExpense"]),
    ("noninterest_income", "us-gaap", "USD", ["NoninterestIncome"]),
    ("cfo", "us-gaap", "USD", [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]),
    ("capex", "us-gaap", "USD", [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ]),
    ("dna", "us-gaap", "USD", [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ]),
    # Share-count fallback for filers that omit the dei cover-page tag. Diluted
    # weighted-average is a duration concept, hence its place among the flows.
    ("shares_diluted", "us-gaap", "shares", [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ]),
]

INSTANT_CONCEPTS: list[tuple[str, str, str, list[str]]] = [
    ("assets", "us-gaap", "USD", ["Assets"]),
    ("equity", "us-gaap", "USD", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
    ("cash", "us-gaap", "USD", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ]),
    ("debt_lt", "us-gaap", "USD", [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "DebtLongtermAndShorttermCombinedAmount",
        "DebtInstrumentCarryingAmount",
    ]),
    ("debt_st", "us-gaap", "USD", [
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "OtherShortTermBorrowings",
    ]),
    # Real-estate carrying value, for a REIT loan-to-value proxy. Only REITs tag it
    # (8/12 of the large REITs; 0/11 banks), so it doubles as a sector sanity check.
    ("real_estate", "us-gaap", "USD", [
        "RealEstateInvestmentPropertyNet",
        "RealEstateInvestmentPropertyAtCost",
    ]),
    ("shares", "dei", "shares", ["EntityCommonStockSharesOutstanding"]),
    ("shares_common", "us-gaap", "shares", [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ]),
]


@dataclass
class Fact:
    """One XBRL fact, with the provenance needed to cite it."""

    value: float
    end: str
    accn: str = ""
    tag: str = ""
    period: str = ""


@dataclass
class CompanyFacts:
    cik: int
    ticker: str
    facts: dict[str, Fact] = field(default_factory=dict)
    net_income_history: list[float] = field(default_factory=list)

    def val(self, key: str, default: float | None = None) -> float | None:
        f = self.facts.get(key)
        return f.value if f else default


# ---------------------------------------------------------------------------
# period helpers — derived from the run date so the pipeline does not rot
# ---------------------------------------------------------------------------
def annual_periods(today: date | None = None, count: int = 3) -> list[str]:
    """Most recent plausibly-filed calendar years, newest first.

    Annual reports land well after year end, so the current year is never
    complete. We start at year-1 and walk backwards.
    """
    y = (today or date.today()).year
    return [f"CY{y - i}" for i in range(1, count + 1)]


def instant_periods(today: date | None = None, count: int = 4) -> list[str]:
    """Most recent balance-sheet dates, newest first (e.g. CY2026Q1I)."""
    d = today or date.today()
    # A quarter's facts only appear once the 10-Q is filed (~40 days), so step
    # back one full quarter before we start looking.
    q = (d.month - 1) // 3
    y = d.year
    out = []
    for _ in range(count):
        q -= 1
        if q < 0:
            q, y = 3, y - 1
        out.append(f"CY{y}Q{q + 1}I")
    return out


def quarter_periods(today: date | None = None) -> tuple[str, str]:
    """(most recent reported quarter, same quarter one year earlier).

    Used to roll a stale annual figure forward into a true trailing-twelve-month
    number: ``TTM = FY + Q_latest - Q_latest_prior_year``.
    """
    latest_instant = instant_periods(today, 1)[0]        # e.g. CY2026Q1I
    stem = latest_instant[:-1]                            # CY2026Q1
    year = int(stem[2:6])
    return stem, f"CY{year - 1}{stem[6:]}"


# ---------------------------------------------------------------------------
# frame fetching
# ---------------------------------------------------------------------------
def _frame(tax: str, concept: str, unit: str, period: str) -> dict[int, Fact]:
    """One SEC frame as {cik: Fact}. Missing frames resolve to an empty dict."""
    url = FRAMES.format(tax=tax, concept=concept, unit=unit, period=period)
    try:
        payload = cached_json(f"edgar/{tax}_{concept}_{unit}_{period}", url, **_sec_kw())
    except Exception:
        return {}
    out: dict[int, Fact] = {}
    for row in payload.get("data", []):
        cik = row.get("cik")
        val = row.get("val")
        if cik is None or val is None:
            continue
        out[int(cik)] = Fact(
            value=float(val), end=str(row.get("end", "")),
            accn=str(row.get("accn", "")), tag=concept, period=period,
        )
    return out


def _resolve(tax: str, unit: str, tags: list[str], periods: list[str]) -> dict[int, Fact]:
    """Merge frames across tags and periods, preferring newer periods and earlier tags.

    A company absent from the newest period (late filer, non-calendar fiscal year)
    is filled in from an older one rather than dropped. Coverage beats recency here:
    a stale-but-present ROIC still ranks correctly against its sector, whereas a
    null silently becomes a false AVOID — the exact failure this rebuild exists to fix.
    """
    merged: dict[int, Fact] = {}
    for period in periods:
        for tag in tags:
            for cik, fact in _frame(tax, tag, unit, period).items():
                if cik not in merged:
                    merged[cik] = fact
    return merged


def _resolve_flow(tax: str, unit: str, tags: list[str], annuals: list[str],
                  q_now: str, q_prior: str) -> dict[int, Fact]:
    """Trailing-twelve-month value per company, computed within a single XBRL tag.

    Tag consistency is load-bearing. ``TTM = FY + Q_latest - Q_prior_year`` is only
    valid if all three legs measure the same thing, and filers routinely tag the same
    concept differently across periods. Resolving each leg independently across a
    merged tag list can subtract ``Revenues`` from
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` and produce a plausible
    number that is quietly nonsense — the kind of error that survives review precisely
    because it does not look like an error.

    So: pick the first tag with an annual figure for that company, then take all three
    legs from that tag alone.
    """
    frames = {
        tag: {p: _frame(tax, tag, unit, p) for p in (*annuals, q_now, q_prior)}
        for tag in tags
    }
    out: dict[int, Fact] = {}
    for tag in tags:
        per_tag = frames[tag]
        for period in annuals:
            for cik, fy in per_tag[period].items():
                if cik in out:
                    continue
                out[cik] = _ttm(fy, per_tag[q_now].get(cik), per_tag[q_prior].get(cik))
    return {c: f for c, f in out.items() if f is not None}


def cik_map() -> dict[str, int]:
    """{TICKER: cik} from SEC's canonical ticker file."""
    payload = cached_json("edgar/company_tickers", TICKERS_URL, ttl_hours=168, **_sec_kw())
    rows = payload.values() if isinstance(payload, dict) else payload
    out: dict[str, int] = {}
    for row in rows:
        t = str(row.get("ticker", "")).upper()
        if t:
            out[t] = int(row["cik_str"])
    return out


def _ttm(fy: Fact | None, q_now: Fact | None, q_prior: Fact | None) -> Fact | None:
    """Roll a fiscal-year figure forward to trailing-twelve-months where possible.

    Requires all three legs; otherwise returns the annual figure unchanged so a
    partially-covered company still gets a usable (if slightly stale) number.
    """
    if fy is None:
        return None
    if q_now is None or q_prior is None:
        return fy
    return Fact(
        value=fy.value + q_now.value - q_prior.value,
        end=q_now.end, accn=q_now.accn, tag=fy.tag, period=f"TTM@{q_now.period}",
    )


def load(tickers: list[str], today: date | None = None) -> dict[str, CompanyFacts]:
    """Fundamentals for `tickers`, keyed by ticker.

    Roughly 60-90 SEC requests total regardless of universe size — the cost is
    per-concept, not per-company.
    """
    cmap = cik_map()
    wanted = {}
    for t in tickers:
        cik = cmap.get(t.upper()) or cmap.get(t.upper().replace(".", "-"))
        if cik is not None:
            wanted[cik] = t
    if not wanted:
        return {}

    annuals = annual_periods(today)
    instants = instant_periods(today)
    q_now, q_prior = quarter_periods(today)

    out = {t: CompanyFacts(cik=c, ticker=t) for c, t in wanted.items()}
    by_cik = {c: out[t] for c, t in wanted.items()}

    for metric, tax, unit, tags in FLOW_CONCEPTS:
        resolved = _resolve_flow(tax, unit, tags, annuals, q_now, q_prior)
        for cik, comp in by_cik.items():
            fact = resolved.get(cik)
            if fact:
                comp.facts[metric] = fact

    for metric, tax, unit, tags in INSTANT_CONCEPTS:
        resolved = _resolve(tax, unit, tags, instants)
        for cik, comp in by_cik.items():
            fact = resolved.get(cik)
            if fact:
                comp.facts[metric] = fact

    # Earnings stability needs a multi-year series, so net income gets its own
    # pass across a longer window of annual frames.
    ni_tags = next(tags for m, _, _, tags in FLOW_CONCEPTS if m == "net_income")
    history: dict[int, list[float]] = {c: [] for c in by_cik}
    for period in annual_periods(today, count=6):
        frame = _resolve("us-gaap", "USD", ni_tags, [period])
        for cik in by_cik:
            fact = frame.get(cik)
            if fact:
                history[cik].append(fact.value)
    for cik, comp in by_cik.items():
        comp.net_income_history = history[cik]

    return out


# ---------------------------------------------------------------------------
# derived metrics
# ---------------------------------------------------------------------------
def _safe_div(num: float | None, den: float | None, floor: float = 1.0) -> float | None:
    if num is None or den is None or abs(den) < floor:
        return None
    return num / den


def earnings_stability(series: list[float]) -> float | None:
    """1 - coefficient of variation of annual net income, clamped to [0, 1].

    A company earning 10, 11, 10, 12 scores near 1. One earning 10, -5, 30, 2
    scores near 0. Fewer than three observations is not a distribution, so we
    return None rather than a number that looks more confident than it is.
    """
    vals = [v for v in series if v is not None]
    if len(vals) < 3:
        return None
    mean = statistics.fmean(vals)
    if abs(mean) < 1.0:
        return 0.0
    cv = statistics.pstdev(vals) / abs(mean)
    return max(0.0, min(1.0, 1.0 - cv))


def derive(comp: CompanyFacts, market_cap: float | None) -> dict:
    """Turn raw facts into the ratio set the model consumes.

    Every output is either a real number or None. None means "we do not know",
    and the model handles it explicitly — it is never coerced to zero, because a
    zero ROIC is a *claim* about a company and an absent one is not.
    """
    revenue = comp.val("revenue")
    gross_profit = comp.val("gross_profit")
    if gross_profit is None and revenue is not None:
        cost = comp.val("cost_of_revenue")
        if cost is not None:
            gross_profit = revenue - cost
    operating_income = comp.val("operating_income")
    cfo = comp.val("cfo")
    capex = comp.val("capex")
    dna = comp.val("dna")
    equity = comp.val("equity")
    cash = comp.val("cash")
    debt = (comp.val("debt_lt") or 0.0) + (comp.val("debt_st") or 0.0)
    has_debt_tag = comp.facts.get("debt_lt") is not None or comp.facts.get("debt_st") is not None

    gross_margin = _safe_div(gross_profit, revenue)

    # Invested capital = interest-bearing debt + book equity - excess cash.
    #
    # Guard the denominator. Companies that have bought back stock aggressively can
    # carry negative book equity, and a small or negative invested-capital base turns
    # ROIC into a meaningless number with a confident sign — Booking Holdings scored
    # -92% before this guard, which would have ranked a highly profitable business at
    # the bottom of its sector. An unmeasurable ratio must read as unmeasured.
    invested = None
    if equity is not None and has_debt_tag:
        base = debt + equity - (cash or 0.0)
        floor = max(1e6, 0.05 * abs(revenue)) if revenue else 1e6
        if base > floor:
            invested = base
    nopat = operating_income * (1.0 - TAX_RATE) if operating_income is not None else None
    roic = _safe_div(nopat, invested, floor=1e6)

    fcf = (cfo - capex) if (cfo is not None and capex is not None) else None
    fcf_yield = _safe_div(fcf, market_cap, floor=1e6)

    ebitda = (operating_income + dna) if (operating_income is not None and dna is not None) else None
    net_debt = (debt - (cash or 0.0)) if has_debt_tag else None
    net_debt_ebitda = _safe_div(net_debt, ebitda, floor=1e6)

    # --- sector-specific inputs -------------------------------------------------
    # Computed for every filer; only consumed by the sector profile that needs them.
    # A company that does not tag the underlying concepts simply gets None here,
    # which the model treats as unobserved rather than as a zero.
    net_income_common = comp.val("net_income_common") or comp.val("net_income")
    equity = comp.val("equity")
    assets = comp.val("assets")

    # Banks: return on equity, not return on invested capital. "Invested capital" is
    # not a meaningful denominator for a balance sheet whose liabilities are deposits.
    roe = _safe_div(net_income_common, equity, floor=1e6)

    # Banks: cost per dollar of revenue produced. Lower is better, so it is ranked
    # inverted. This is the single most-watched bank efficiency measure.
    nii = comp.val("net_interest_income")
    noninterest_income = comp.val("noninterest_income")
    noninterest_expense = comp.val("noninterest_expense")
    bank_revenue = None
    if nii is not None or noninterest_income is not None:
        bank_revenue = (nii or 0.0) + (noninterest_income or 0.0)
    efficiency_ratio = _safe_div(noninterest_expense, bank_revenue, floor=1e6)

    # Banks: equity to assets. Regulatory CET1 is NOT obtainable — banks tag capital
    # ratios dimensionally (consolidated vs bank-level, Standardized vs Advanced) and
    # the bulk frames API drops dimensional facts, so
    # CommonEquityTierOneCapitalToRiskWeightedAssets returns zero filers universe-wide.
    # Equity/assets is a plain, fully-observable capital cushion, and it is labelled as
    # a proxy rather than presented as a regulatory ratio.
    equity_to_assets = _safe_div(equity, assets, floor=1e6)

    # REITs: GAAP net income is depressed by property depreciation that does not
    # reflect economic decline, which is why the industry reports FFO. The exact
    # NAREIT definition also removes gains on property sales, but that adjustment is
    # not tagged (0/12 of the large REITs), so this is an APPROXIMATION and is
    # labelled as one everywhere it surfaces.
    ffo = None
    if net_income_common is not None and dna is not None:
        ffo = net_income_common + dna
    ffo_yield = _safe_div(ffo, market_cap, floor=1e6)

    # REITs: leverage against the asset base. Assets is used rather than real-estate
    # carrying value because Assets is tagged by 12/12 large REITs and the real-estate
    # concepts by 8/12; a better-covered denominator beats a more precise one that is
    # missing a third of the time.
    debt_to_assets = _safe_div(debt if has_debt_tag else None, assets, floor=1e6)

    return {
        "roic": roic,
        "fcf_yield": fcf_yield,
        "gross_margin": gross_margin,
        "net_debt_ebitda": net_debt_ebitda,
        "earnings_stability": earnings_stability(comp.net_income_history),
        "roe": roe,
        "efficiency_ratio": efficiency_ratio,
        "equity_to_assets": equity_to_assets,
        "ffo": ffo,
        "ffo_yield": ffo_yield,
        "debt_to_assets": debt_to_assets,
        "cfo_yield": _safe_div(cfo, market_cap, floor=1e6),
        "real_estate": comp.val("real_estate"),
        "shares": (comp.val("shares") or comp.val("shares_common")
                   or comp.val("shares_diluted")),
        "revenue_ttm": revenue,
        "net_income_ttm": comp.val("net_income"),
        "fcf_ttm": fcf,
        "ebitda_ttm": ebitda,
        "as_of": {k: {"end": f.end, "accn": f.accn, "tag": f.tag} for k, f in comp.facts.items()},
    }
