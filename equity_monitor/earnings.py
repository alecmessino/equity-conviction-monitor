"""The earnings calendar, derived rather than bought.

There is no keyless forward earnings calendar. Vendors sell one; SEC does not publish
one, because a company is under no obligation to announce its reporting date in a
structured filing. So this module does not pretend to have one. It measures what SEC
*does* publish — when each filer actually reported, every quarter, for years — and
projects the next date from that filer's own cadence.

The distinction that matters, and the one every row carries:

  **confirmed** — a filing has landed. This is a fact with an accession number behind
  it: the 8-K carrying the results release, or the 10-Q/10-K itself.

  **estimated** — no filing yet. The date is this filer's last period end plus the lag
  it has historically taken to report, and it is wrong by a few days routinely.

Mixing those two into one column is how a calendar becomes a liability: a desk that
cannot tell a confirmed date from a projection will schedule around a projection.

Cost. The bulk history comes from EDGAR's own index files, which list every filing made
in a period — one request covers every filer at once, the same property that makes the
``frames`` API affordable for fundamentals. Completed quarters are immutable and cached
indefinitely; the current quarter is assembled from daily index files, each of which is
also immutable once the day is over. There is no per-ticker loop anywhere in this
module, and on a warm cache a run fetches exactly one file: today's.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

from .sources._http import CACHE_DIR, fetch
from .sources.edgar import _sec_kw, cik_map

QUARTERLY = "https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"
DAILY = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{ymd}.idx"

# The forms that carry a reported quarter. 8-K is where the results release lands,
# typically days before the 10-Q that follows it; both are kept so the calendar can
# show the announcement rather than the statutory filing that trails it.
PERIODIC = ("10-Q", "10-K")
RESULTS = ("8-K",)
KEEP = set(PERIODIC) | set(RESULTS)

# A quarter of a fiscal year, in days. Filers drift a few days around it; the drift is
# absorbed by rolling the projection forward until it lands in the future.
QUARTER_DAYS = 91
# Outside this band a "lag" is not a reporting lag — it is a restatement, an amended
# filing, or a period end we mis-paired — and it would poison the median.
LAG_MIN, LAG_MAX = 5, 120
# How far ahead the calendar projects. Past two quarters the estimate is dominated by
# accumulated cadence drift and stops being worth printing.
HORIZON_DAYS = 190
# A filing this fresh is this cycle's print rather than something to project. It stays
# on the calendar briefly so the week's reporters are visible instead of disappearing.
CONFIRM_DAYS = 6
# How far past the last periodic filing a lone 8-K has to sit before it is read as the
# next quarter's results release rather than an event inside the current one.
STANDALONE_8K_DAYS = 60


# ---------------------------------------------------------------------------
# EDGAR index files
# ---------------------------------------------------------------------------
def _quarter(d: date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def _quarter_start(y: int, q: int) -> date:
    return date(y, (q - 1) * 3 + 1, 1)


def _prev_quarter(y: int, q: int) -> tuple[int, int]:
    return (y - 1, 4) if q == 1 else (y, q - 1)


def _parse_idx(text: str) -> list[tuple[str, int, str]]:
    """(form, cik, filed) for every filing of interest in one index file.

    Positional from the *end* of the line, not by regex. The two index files EDGAR
    publishes do not agree on the date format — the quarterly writes ``2026-05-07`` and
    the daily writes ``20260814`` — so a search for an ISO date finds nothing in a daily
    file and then matches inside the accession number in the path instead, which yields
    a plausible-looking date and a four-digit CIK and is wrong on every row. It failed
    silently: the calendar simply had no filings newer than the last completed quarter.

    Company names contain spaces and the form types kept here do not, so splitting on
    whitespace and reading form from the front and path/date/CIK from the back is stable
    across both formats and across every era EDGAR has published.
    """
    out: list[tuple[str, int, str]] = []
    for line in text.splitlines():
        if not line or line[0] in "- " or line.startswith("Form Type"):
            continue
        parts = line.split()
        if len(parts) < 4 or parts[0] not in KEEP:
            continue
        cik, filed = parts[-3], parts[-2]
        if not cik.isdigit():
            continue
        if re.fullmatch(r"\d{8}", filed):
            filed = f"{filed[:4]}-{filed[4:6]}-{filed[6:]}"
        elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", filed):
            continue
        out.append((parts[0], int(cik), filed))
    return out


def _cached_index(key: str, url: str, *, ttl_hours: float) -> list[tuple[str, int, str]]:
    """Fetch and filter one index file, caching the *filtered* rows.

    The raw quarterly file is 25–55 MB and 99% of it is forms this module does not read.
    Caching the parsed remainder keeps a warm run to a few megabytes on disk and makes
    a re-parse free.
    """
    path = os.path.join(CACHE_DIR, "edgar", "idx", key + ".json")
    if os.path.exists(path) and ttl_hours > 0:
        age_h = (os.path.getmtime(path) and
                 (datetime.now().timestamp() - os.path.getmtime(path)) / 3600.0)
        if age_h < ttl_hours:
            try:
                with open(path) as fh:
                    return [tuple(r) for r in json.load(fh)]
            except Exception:
                pass  # corrupt entry — fall through and re-fetch
    body = fetch(url, **_sec_kw())
    rows = _parse_idx(body.decode("latin-1", "replace"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh, separators=(",", ":"))
    os.replace(tmp, path)
    return rows


def filing_index(today: date | None = None, quarters: int = 2,
                 on_progress=None) -> dict[int, list[tuple[str, str]]]:
    """{cik: [(filed, form), ...]} newest first, across the recent filing history.

    Completed quarters come from the quarterly index and never change, so they are
    cached without expiry. The current quarter is assembled from daily files: each is
    immutable once the day is over, so only today's is ever re-fetched.
    """
    today = today or date.today()
    rows: list[tuple[str, int, str]] = []

    y, q = _quarter(today)
    done: list[tuple[int, int]] = []
    cy, cq = y, q
    for _ in range(max(0, quarters)):
        cy, cq = _prev_quarter(cy, cq)
        done.append((cy, cq))
    for i, (qy, qq) in enumerate(done):
        if on_progress:
            on_progress(f"index {qy}Q{qq}", i, len(done) + 1)
        try:
            rows += _cached_index(f"q{qy}Q{qq}", QUARTERLY.format(y=qy, q=qq),
                                  ttl_hours=24 * 365)
        except Exception:
            continue  # one missing quarter narrows the history, it does not break it

    # Current quarter, day by day. Weekends and holidays 404 and are simply absent.
    d = _quarter_start(y, q)
    days = 0
    while d <= today:
        if d.weekday() < 5:
            ymd = d.strftime("%Y%m%d")
            # Today's file is still being written; everything older is final.
            ttl = 4 if d == today else 24 * 365
            try:
                rows += _cached_index(f"d{ymd}", DAILY.format(y=y, q=q, ymd=ymd),
                                      ttl_hours=ttl)
                days += 1
            except Exception:
                pass
        d += timedelta(days=1)
    if on_progress:
        on_progress(f"index current quarter ({days}d)", len(done), len(done) + 1)

    by_cik: dict[int, list[tuple[str, str]]] = {}
    for form, cik, filed in rows:
        by_cik.setdefault(cik, []).append((filed, form))
    for v in by_cik.values():
        v.sort(reverse=True)
    return by_cik


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    symbol: str
    name: str = ""
    sector: str = ""
    cik: int | None = None
    conviction: int | None = None
    signal: str = ""
    weight: float = 0.0
    in_book: bool = False
    # what actually happened
    last_reported: str | None = None      # the announcement, or the periodic filing
    last_form: str | None = None
    last_period_end: str | None = None
    lag_days: int | None = None
    # what is projected
    date: str | None = None
    state: str = "unknown"                # confirmed | estimated | unknown
    band_days: int | None = None
    basis: str = ""
    days_away: int | None = None


def _parse(d: str | None) -> date | None:
    try:
        return datetime.strptime(d, "%Y-%m-%d").date() if d else None
    except Exception:
        return None


def _period_end(row: dict) -> date | None:
    """The period the ledger's facts describe: the date most of them agree on.

    Not ``max(end)``. ``EntityCommonStockSharesOutstanding`` is a dei cover-page fact
    dated when the filing was *prepared*, which for NetApp is 2026-05-28 against a
    fiscal quarter that ended 2026-04-24 — so the maximum is the cover date, sitting a
    few days before the filing itself. Pairing that with the filing measured
    cover-date-to-filing rather than period-end-to-filing and reported an 8-day median
    reporting lag for a 10-Q that takes about a month. Every raw lag came out at 8,
    which is what a distribution looks like when it is measuring the wrong quantity.

    The mode is robust without a hand-maintained tag allowlist that would rot the first
    time a new concept is added: sixteen of NetApp's seventeen facts carry the real
    period end and one does not.
    """
    ends: list[date] = []
    for fact in (row.get("as_of") or {}).values():
        if not fact:
            continue
        # dei concepts describe the filing entity, never a reporting period.
        if str(fact.get("tag") or "").startswith("Entity"):
            continue
        d = _parse(fact.get("end"))
        if d:
            ends.append(d)
    if not ends:
        return None
    counts = Counter(ends)
    top = max(counts.values())
    return max(d for d, c in counts.items() if c == top)


def _announcements(filings: list[tuple[str, str]]) -> list[tuple[date, str]]:
    """Every reported quarter in the window, newest first.

    Companies put the numbers out on an 8-K and file the 10-Q days later. The 8-K is the
    date a desk cares about, so where one sits within a week before the periodic filing
    it is treated as the announcement for that quarter and the 10-Q behind it is dropped.
    """
    periodic = sorted(((_parse(d), f) for d, f in filings if f in PERIODIC),
                      reverse=True)
    periodic = [(d, f) for d, f in periodic if d]
    eights = sorted((_parse(d) for d, f in filings if f in RESULTS), reverse=True)
    eights = [d for d in eights if d]
    out: list[tuple[date, str]] = []
    used: set[date] = set()
    for pd_, pf in periodic:
        near = [d for d in eights if -7 <= (d - pd_).days <= 1]
        if near:
            d = max(near)
            out.append((d, "8-K"))
            used.add(d)
        else:
            out.append((pd_, pf))
    # A results release that has not been followed by its 10-Q yet. This is the whole
    # point of watching 8-K: the numbers go out days before the statutory filing, and
    # anchoring only on periodic filings meant a company that reported on Monday still
    # showed a projected date until the 10-Q landed the following week.
    #
    # Only counted when it stands far enough past the last periodic filing to be the
    # next quarter's print. An 8-K is also filed for officer changes, acquisitions and
    # covenant amendments, and promoting one of those would confirm a date that never
    # happened — a worse failure than leaving the estimate up.
    newest = periodic[0][0] if periodic else None
    for d in eights:
        if d in used:
            continue
        if newest is None or (d - newest).days >= STANDALONE_8K_DAYS:
            out.append((d, "8-K"))
    out.sort(reverse=True)
    return out


def _measure_lag(period_end: date | None, anns: list[tuple[date, str]],
                 index_start: date | None = None) -> int | None:
    """Days from a period end to the filing that reported it.

    Paired against the *first* announcement after the period end, not the most recent
    one. The ledger's fundamentals come from SEC's ``frames`` API, which resolves the
    most recent complete frame and therefore trails the filings by up to two quarters —
    so pairing a period end with the newest filing measures the staleness of the
    fundamentals rather than the filer's reporting lag. That error is silent and it is
    directional: it made the first cut of this module report a 12-day median for a
    filing that takes about a month, on 113 usable pairs out of 1014.
    """
    if not period_end:
        return None
    # The filing that reported this period has to be inside the window we indexed. For a
    # period that ended before the window opens, the earliest filing we can see is the
    # *next* quarter's, and pairing them measures a quarter plus a lag: Nordson came out
    # at 110 days and Deere at 117, both of which passed the sanity band and dragged the
    # projection with them. November-fiscal filers fail this test most often, so they
    # fall back to the pooled median and the row says which it used.
    if index_start and period_end < index_start:
        return None
    after = [d for d, _ in anns if (d - period_end).days >= LAG_MIN]
    if not after:
        return None
    lag = (min(after) - period_end).days
    return lag if LAG_MIN <= lag <= LAG_MAX else None


def build(rows: list[dict], today: date | None = None,
          quarters: int = 2, on_progress=None) -> dict:
    """The calendar payload, over the scored universe.

    Every entry is either a fact with a filing behind it or a projection that says so.
    Nothing here is back-filled and nothing is bought.
    """
    today = today or date.today()
    scored = [r for r in rows if r.get("conviction") is not None]
    cmap = cik_map()
    index = filing_index(today, quarters=quarters, on_progress=on_progress)
    # The earliest date the index can see a filing on.
    iy, iq = _quarter(today)
    for _ in range(max(1, quarters)):
        iy, iq = _prev_quarter(iy, iq)
    index_start = _quarter_start(iy, iq)

    # Universe lag first: most names contribute one (period end -> filed) observation,
    # and a single observation is not a distribution. The pooled median is what a name
    # with no usable pair of its own falls back to, and it is reported so a reader can
    # see which entries are leaning on it.
    lags: list[int] = []
    prepared: list[tuple[dict, int | None, list[tuple[date, str]], date | None, int | None]] = []
    for r in scored:
        sym = str(r.get("symbol") or "").upper()
        cik = cmap.get(sym) or cmap.get(sym.replace(".", "-"))
        anns = _announcements(index.get(cik, [])) if cik else []
        pe = _period_end(r)
        own = _measure_lag(pe, anns, index_start)
        prepared.append((r, cik, anns, pe, own))
        if own is not None:
            lags.append(own)
    median_lag = int(statistics.median(lags)) if lags else 40
    # Median absolute deviation, not standard deviation: the lag distribution has a tail
    # of late filers that would inflate a sigma into a band covering half a quarter.
    band = int(statistics.median([abs(x - median_lag) for x in lags])) if lags else 7
    band = max(2, min(band, 14))

    entries: list[Entry] = []
    unresolved: list[str] = []
    for r, cik, anns, pe, own in prepared:
        sym = str(r.get("symbol") or "").upper()
        e = Entry(
            symbol=sym, name=r.get("name") or "", sector=r.get("sector") or "",
            cik=cik, conviction=r.get("conviction"), signal=r.get("signal") or "",
            weight=float(r.get("weight") or 0.0),
            in_book=bool(r.get("weight")),
            last_period_end=pe.isoformat() if pe else None,
        )
        if not cik:
            unresolved.append(sym)
            e.basis = "no CIK in SEC's ticker file — nothing to project from"
            entries.append(e)
            continue
        if not anns:
            e.basis = (f"no 10-Q or 10-K in the last {quarters} quarters of the EDGAR "
                       f"index — too little history to project a cadence")
            entries.append(e)
            continue
        last_d, last_f = anns[0]
        e.last_reported, e.last_form = last_d.isoformat(), last_f
        lag = own if own is not None else median_lag
        e.lag_days = lag
        src = "its own measured lag" if own is not None else f"the universe median lag ({median_lag}d)"

        # A filing this recent IS this cycle's print. Nothing to project: the row is a
        # fact with a form behind it, and it stays on the calendar for a few days so the
        # week's reporters are visible rather than vanishing the moment they report.
        if (today - last_d).days <= CONFIRM_DAYS:
            e.date, e.state, e.band_days = last_d.isoformat(), "confirmed", 0
            e.basis = (f"{last_f} filed {last_d.isoformat()} — a filing that has landed, "
                       f"not a projection")
        else:
            # Anchored on the most recent announcement rather than on the ledger's period
            # end. Both describe the same quarter, but the announcement is fresh where
            # `as_of` trails the filings by up to two quarters — anchoring on the stale
            # one and stepping forward compounds that error into the projection.
            implied_end = last_d - timedelta(days=lag)
            nxt, guard = implied_end, 0
            while nxt + timedelta(days=lag) <= today and guard < 8:
                nxt += timedelta(days=QUARTER_DAYS)
                guard += 1
            projected = nxt + timedelta(days=lag)
            e.date, e.state, e.band_days = projected.isoformat(), "estimated", band
            e.basis = (f"last reported {last_d.isoformat()} on a {last_f}, {lag} days after "
                       f"a period end, from {src} — one fiscal quarter on from there")
        d = _parse(e.date)
        e.days_away = (d - today).days if d else None
        entries.append(e)

    entries.sort(key=lambda x: (x.date or "9999-99-99", -(x.conviction or 0)))
    horizon = [e for e in entries
               if e.days_away is not None and -CONFIRM_DAYS <= e.days_away <= HORIZON_DAYS]
    book = [e for e in horizon if e.in_book]
    return {
        "as_of": today.isoformat(),
        "source": "SEC EDGAR full-index (completed quarters) + daily-index (current quarter)",
        "quarters_indexed": quarters,
        "index_from": index_start.isoformat(),
        "ciks_in_index": len(index),
        "universe": len(scored),
        "resolved": sum(1 for e in entries if e.cik),
        "unresolved": unresolved[:40],
        "unresolved_n": len(unresolved),
        "projected": sum(1 for e in entries if e.state == "estimated"),
        "confirmed": sum(1 for e in entries if e.state == "confirmed"),
        "median_lag_days": median_lag,
        "lag_observations": len(lags),
        "band_days": band,
        "horizon_days": HORIZON_DAYS,
        "in_horizon": len(horizon),
        "in_book": len(book),
        "entries": [asdict(e) for e in entries],
        "basis": (
            f"Dates are derived from EDGAR's own filing index, not from a vendor "
            f"calendar — SEC does not publish forward earnings dates and no company is "
            f"obliged to file one. A **confirmed** row is a filing that has landed and "
            f"carries the form that reported it. An **estimated** row is the filer's "
            f"last period end plus the lag it has historically taken to report: "
            f"{median_lag} days at the universe median across {len(lags)} observations, "
            f"±{band} at the median absolute deviation. Estimates are routinely wrong by "
            f"a few days and must not be scheduled around as if they were not."
        ),
    }


def write(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return path
