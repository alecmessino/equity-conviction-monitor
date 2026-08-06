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
import json
import sys

# Each threshold below corresponds to an observed real failure, not a hypothetical.
MIN_ROWS = 20
MIN_DISPERSION = 5.0          # v2 shipped ~0.2: every name scored 0 or 1
MIN_DISTINCT_TIERS = 3        # v2 populated exactly one tier (AVOID)
MIN_SECTOR_COVERAGE = 0.90    # v2: 39 of 48 names had no sector at all
MCAP_LOW, MCAP_HIGH = 1e8, 2e13   # v2 published GOOGL at 4.4e18 (a 1e6 unit error)
MIN_FIELD_COVERAGE = {
    "roic": 0.70,             # v2: 0.00
    "fcf_yield": 0.70,        # v2: 0.00
    "market_cap": 0.90,       # v2: 0.19
    "rs_blend": 0.95,         # v2: 0.00 — momentum was constant across the universe
    "drawdown_52w": 0.90,
}


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

    if not payload.get("as_of"):
        fail("no as_of timestamp")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="ledger/index.json")
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS)
    args = ap.parse_args()

    try:
        with open(args.path) as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"FAIL  cannot read {args.path}: {exc}")
        return 2

    problems = check(payload, args.min_rows)
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
