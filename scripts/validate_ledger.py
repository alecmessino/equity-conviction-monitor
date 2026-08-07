#!/usr/bin/env python3
"""Anti-degeneracy gate for ledger/index.json.

This exists because the previous pipeline ran green for weeks while publishing a
board on which every one of 48 names scored 0 and read AVOID. Nothing threw. The
workflow reported success. The site deployed. The only thing wrong was the numbers,
and nothing was looking at the numbers.

A build that produces a *structurally valid* ledger full of meaningless values is a
failed build, and this script is what says so. Run it between the data build and the
deploy; a non-zero exit must block publication.

    python scripts/validate_ledger.py ledger/index.json
    python scripts/validate_ledger.py ledger/index.json --min-rows 400
"""
from __future__ import annotations

import argparse
import os
import json
import sys

# Each threshold below corresponds to an observed real failure, not a hypothetical.
MIN_ROWS = 20
MIN_DISPERSION = 5.0          # v2 shipped ~0.2: every name scored 0 or 1
MIN_DISTINCT_TIERS = 3        # v2 populated exactly one tier (AVOID)
MIN_SECTOR_COVERAGE = 0.90    # v2: 39 of 48 names had no sector at all
MCAP_LOW, MCAP_HIGH = 1e8, 2e13   # v2 published GOOGL at 4.4e18 (a 1e6 unit error)
# Absolute floors are set to catch *structural* breakage (a field going to zero),
# not to demand perfection. They sit well below observed coverage on a full Russell
# 1000 run so a legitimate build is never blocked: at 1,013 names the pipeline
# reports roic 64%, fcf_yield 82%, market_cap 97%. Fundamentals coverage is
# genuinely lower than large-cap-only, because banks do not report gross margin and
# small caps tag inconsistently — that is real, not a defect.
MIN_FIELD_COVERAGE = {
    "roic": 0.40,             # v2: 0.00
    "fcf_yield": 0.55,        # v2: 0.00
    "market_cap": 0.85,       # v2: 0.19
    "rs_blend": 0.95,         # v2: 0.00 — momentum was constant across the universe
    "drawdown_52w": 0.90,
}

# Run-over-run drop that fails the build. Catches what a loose floor cannot: a vendor
# renaming one tag and taking a field from 85% to 55% — clearly broken, comfortably
# above any floor safe enough to leave in place.
COVERAGE_REGRESSION = 0.15

# A sector profile exists to stop that sector being scored on imputed values. If its
# own inputs resolve for less than this share of the sector, the profile is not doing
# the job it was introduced for and the build should say so.
PROFILE_INPUT_COVERAGE = 0.75


class Failure(Exception):
    pass


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def check(payload: dict, min_rows: int = MIN_ROWS) -> list[str]:
    """Return a list of failure messages. Empty means the ledger is publishable."""
    problems: list[str] = []
    rows = payload.get("all") or []

    def fail(msg: str) -> None:
        problems.append(msg)

    if len(rows) < min_rows:
        fail(f"only {len(rows)} scored rows (need >= {min_rows})")
    if not rows:
        return problems

    convictions = [r.get("conviction") for r in rows if r.get("conviction") is not None]
    if len(convictions) < len(rows):
        fail(f"{len(rows) - len(convictions)} rows carry no conviction at all")
    if not convictions:
        return problems

    spread = _stdev(convictions)
    if spread < MIN_DISPERSION:
        fail(f"conviction dispersion {spread:.2f} < {MIN_DISPERSION} — the model is "
             f"not discriminating between names (range "
             f"{min(convictions)}-{max(convictions)})")

    tiers = {r.get("signal") for r in rows if r.get("signal")}
    if len(tiers) < MIN_DISTINCT_TIERS:
        fail(f"only {len(tiers)} signal tier(s) populated ({sorted(tiers)}) — "
             f"need >= {MIN_DISTINCT_TIERS}")

    # A pillar that is identical for every name carries no information, which is
    # exactly how v2's confirmation term behaved once relative strength collapsed to 0.
    for pillar in ("q", "c", "r"):
        vals = [r.get(pillar) for r in rows if r.get(pillar) is not None]
        if len(vals) > 5 and _stdev(vals) < 0.01:
            fail(f"pillar '{pillar}' is constant across the universe "
                 f"(value {vals[0]}) — it is contributing nothing to the score")

    caps = [r.get("market_cap") for r in rows if r.get("market_cap")]
    bad_caps = [c for c in caps if not (MCAP_LOW <= c <= MCAP_HIGH)]
    if bad_caps:
        fail(f"{len(bad_caps)} market caps outside [{MCAP_LOW:.0e}, {MCAP_HIGH:.0e}] "
             f"— e.g. {max(bad_caps):.3e} (unit error?)")

    with_sector = sum(1 for r in rows if r.get("sector"))
    if with_sector / len(rows) < MIN_SECTOR_COVERAGE:
        fail(f"sector coverage {with_sector / len(rows):.0%} < "
             f"{MIN_SECTOR_COVERAGE:.0%} — filters and the sector map will render empty")

    coverage = payload.get("coverage") or {}
    for field, floor in MIN_FIELD_COVERAGE.items():
        got = coverage.get(field)
        if got is None:
            observed = sum(1 for r in rows if r.get(field) is not None)
            got = observed / len(rows)
        if got < floor:
            fail(f"coverage[{field}] = {got:.0%} < {floor:.0%}")

    # Sector profiles: check coverage where it matters. A profile input is expected to
    # be sparse universe-wide (only banks report noninterest expense), so a global
    # floor would either be trivially loose or fire constantly. The question that
    # matters is whether the sector scored by a profile actually has the inputs.
    profiles = payload.get("quality_profiles") or {}
    if profiles and rows:
        by_sector: dict[str, list] = {}
        for r in rows:
            by_sector.setdefault(r.get("sector") or "", []).append(r)
        for sector, weights in profiles.items():
            if sector == "default":
                continue
            members = by_sector.get(sector) or []
            if not members:
                continue
            for key in weights:
                got = sum(1 for r in members if r.get(key) is not None) / len(members)
                if got < PROFILE_INPUT_COVERAGE:
                    fail(f"{sector} profile input {key} resolves for {got:.0%} of the "
                         f"sector (need >= {PROFILE_INPUT_COVERAGE:.0%}) — the profile "
                         f"is imputing more than it measures")
            wrong = [r["symbol"] for r in members
                     if r.get("profile") not in (None, sector)]
            if wrong:
                fail(f"{len(wrong)} {sector} names scored under the wrong profile "
                     f"(e.g. {wrong[:3]})")

        # A profile must not leak. Names outside a profile sector must score under
        # the default, or a profile change would silently move the whole board.
        leaked = [r["symbol"] for r in rows
                  if (r.get("sector") or "") not in profiles
                  and r.get("profile") not in (None, "default")]
        if leaked:
            fail(f"{len(leaked)} names outside a profile sector are not on the default "
                 f"profile (e.g. {leaked[:3]})")

    if not payload.get("as_of"):
        fail("no as_of timestamp")
    return problems


def check_regression(payload: dict, previous: dict,
                     drop: float = COVERAGE_REGRESSION) -> list[str]:
    """Compare against the last published ledger.

    The absolute floors above are set low enough to catch structural breakage without
    blocking a legitimate build, which leaves a gap: a vendor renaming one XBRL tag
    might take a field from 85% to 55% — clearly broken, comfortably above any floor
    loose enough to be safe. A run-over-run drop catches that, and it is the shape the
    original failure actually had.
    """
    problems: list[str] = []
    before, after = previous.get("coverage") or {}, payload.get("coverage") or {}
    for field, was in before.items():
        now = after.get(field)
        if now is None:
            problems.append(f"coverage[{field}] disappeared (was {was:.0%})")
        elif was - now > drop:
            problems.append(
                f"coverage[{field}] fell {was:.0%} -> {now:.0%} "
                f"(drop of {was - now:.0%} exceeds {drop:.0%})")
    prev_rows = len(previous.get("all") or [])
    now_rows = len(payload.get("all") or [])
    if prev_rows and now_rows < prev_rows * 0.8:
        problems.append(f"universe shrank {prev_rows} -> {now_rows} names")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="ledger/index.json")
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    ap.add_argument("--previous", help="prior ledger to compare coverage against "
                                       "(CI passes the copy from git HEAD)")
    args = ap.parse_args()

    try:
        with open(args.path) as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"FAIL  cannot read {args.path}: {exc}")
        return 2

    problems = check(payload, args.min_rows)
    if args.previous and os.path.exists(args.previous):
        try:
            with open(args.previous) as fh:
                problems += check_regression(payload, json.load(fh))
        except Exception as exc:
            print(f"note: could not compare against {args.previous}: {exc}")
    rows = payload.get("all") or []
    convictions = [r.get("conviction") for r in rows if r.get("conviction") is not None]
    print(f"ledger:      {args.path}")
    print(f"as_of:       {payload.get('as_of')}  model={payload.get('model_version')}")
    print(f"rows:        {len(rows)} scored, {len(payload.get('benchmarks') or [])} benchmarks")
    if convictions:
        print(f"conviction:  {min(convictions)}-{max(convictions)}  "
              f"dispersion={_stdev(convictions):.1f}  "
              f"mean={sum(convictions) / len(convictions):.1f}")
        tiers: dict[str, int] = {}
        for r in rows:
            tiers[r.get("signal", "?")] = tiers.get(r.get("signal", "?"), 0) + 1
        print("tiers:       " + "  ".join(f"{k}={v}" for k, v in sorted(tiers.items())))
    cov = payload.get("coverage") or {}
    if cov:
        print("coverage:    " + "  ".join(f"{k}={v:.0%}" for k, v in sorted(cov.items())))

    if problems:
        print(f"\nFAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASS  ledger is publishable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
