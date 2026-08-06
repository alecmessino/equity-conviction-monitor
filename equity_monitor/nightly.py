"""Nightly build: universe -> prices + filings -> features -> scores -> ledger.

Writes:
  ledger/index.json          scored universe, run metadata, coverage report
  ledger/history.json        downsampled closes for every name (one fetch for the grid)
  ledger/history/SYM.json    full OHLCV, lazily loaded by the detail view
  ledger/macro.json          FRED regime series

Failure policy: this job would rather ship nothing than ship something wrong. It
prints a coverage report on every run and exits non-zero if the result is degenerate,
because the previous pipeline's defining property was that it "succeeded" for weeks
while publishing a board of zeros.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equity_monitor import features, model, universe as uni
from equity_monitor.sources import edgar, macro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "ledger")
HISTORY = os.path.join(LEDGER, "history")

COVERAGE_FIELDS = [
    "market_cap", "roic", "fcf_yield", "gross_margin", "net_debt_ebitda",
    "earnings_stability", "earnings_yield", "rs_blend", "trend", "adv_usd",
    "vol_1y", "drawdown_52w", "sector",
]

# Portfolio construction: only names the model actually likes are held, weighted by
# conviction above the WATCH boundary and capped so no single position dominates.
# v2 assigned a weight to every name including the ones it rated AVOID, which made the
# "paper index" a market-cap tracker with extra steps.
HOLD_THRESHOLD = 55
WEIGHT_BASE = 40
MAX_WEIGHT = 5.0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _downsample(values: list[float], target: int = 130) -> list[float]:
    if len(values) <= target:
        return [round(v, 4) for v in values]
    step = len(values) / target
    return [round(values[min(len(values) - 1, int(i * step))], 4) for i in range(target)]


def assign_weights(rows: list[dict]) -> None:
    held = [r for r in rows if (r.get("conviction") or 0) >= HOLD_THRESHOLD]
    raw = {r["symbol"]: max(0.0, r["conviction"] - WEIGHT_BASE) for r in held}
    total = sum(raw.values())
    for r in rows:
        r["weight"] = 0.0
    if total <= 0:
        return
    # Cap, then redistribute the overflow across uncapped names until it settles.
    weights = {s: v / total * 100.0 for s, v in raw.items()}
    for _ in range(10):
        excess = sum(max(0.0, w - MAX_WEIGHT) for w in weights.values())
        if excess < 1e-9:
            break
        room = {s: MAX_WEIGHT - w for s, w in weights.items() if w < MAX_WEIGHT}
        room_total = sum(room.values())
        if room_total <= 0:
            break
        weights = {
            s: (MAX_WEIGHT if w >= MAX_WEIGHT else w + excess * room[s] / room_total)
            for s, w in weights.items()
        }
    for r in held:
        r["weight"] = round(weights.get(r["symbol"], 0.0), 3)


def turnover(prev: dict | None, rows: list[dict]) -> dict:
    """One-way turnover and the names entering/leaving the held book."""
    if not prev:
        return {}
    before = {r["symbol"]: r.get("weight", 0.0) for r in prev.get("all", [])
              if r.get("weight")}
    after = {r["symbol"]: r.get("weight", 0.0) for r in rows if r.get("weight")}
    symbols = set(before) | set(after)
    moved = sum(abs(after.get(s, 0.0) - before.get(s, 0.0)) for s in symbols)
    return {
        "one_way_pct": round(moved / 2.0, 2),
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
    }


def build(limit: int | None = None, skip_macro: bool = False,
          skip_fundamentals: bool = False) -> dict:
    members, provenance = uni.load(limit)
    equities = [m for m in members if not m.is_etf]
    print(f"universe: {len(equities)} equities + {len(members) - len(equities)} ETFs "
          f"({provenance})")

    os.makedirs(HISTORY, exist_ok=True)
    symbols = [m.symbol for m in members]
    failures: list[str] = []

    def progress(i: int, total: int, sym: str, ok: bool) -> None:
        if not ok:
            failures.append(sym)
        if i % 25 == 0 or i == total:
            print(f"  prices {i}/{total} ({len(failures)} unavailable)")

    print("fetching prices…")
    bars = features.fetch_bars(symbols, cache_dir=HISTORY, on_progress=progress)
    print(f"prices: {len(bars)}/{len(symbols)} symbols; unavailable: {failures[:8]}")
    if uni.BENCHMARK not in bars:
        raise RuntimeError(f"benchmark {uni.BENCHMARK} unavailable — aborting rather "
                           "than publishing relative strength measured against nothing")

    facts = {}
    if not skip_fundamentals:
        print("fetching SEC filings…")
        scoreable = [m.symbol for m in members if not m.is_etf and m.symbol in bars]
        facts = edgar.load(scoreable)
        print(f"filings: {len(facts)}/{len(scoreable)} companies matched to a CIK")

    rows = features.build(members, bars, fundamentals=facts)
    model.score_rows(rows)
    scored = [r for r in rows if r.get("conviction") is not None]
    scored.sort(key=lambda r: r["conviction"], reverse=True)
    assign_weights(scored)

    cov = features.coverage(rows, COVERAGE_FIELDS)
    print("coverage: " + "  ".join(f"{k}={v:.0%}" for k, v in cov.items()))
    print(f"conviction: n={len(scored)} dispersion={model.dispersion(scored):.1f} "
          f"range={min((r['conviction'] for r in scored), default=0)}–"
          f"{max((r['conviction'] for r in scored), default=0)}")

    # history: one bundled file for the grid, per-symbol OHLCV for the detail view
    bundle = {}
    for sym, b in bars.items():
        bundle[sym] = _downsample(b.close)
        with open(os.path.join(HISTORY, f"{sym}.json"), "w") as fh:
            json.dump(b.to_dict(), fh, separators=(",", ":"))
    with open(os.path.join(LEDGER, "history.json"), "w") as fh:
        json.dump(bundle, fh, separators=(",", ":"))

    if not skip_macro:
        try:
            with open(os.path.join(LEDGER, "macro.json"), "w") as fh:
                json.dump(macro.load(), fh, separators=(",", ":"))
            print("macro: FRED series written")
        except Exception as exc:
            print(f"macro: unavailable ({exc})")

    prev = None
    prev_path = os.path.join(LEDGER, "index.json")
    if os.path.exists(prev_path):
        try:
            with open(prev_path) as fh:
                prev = json.load(fh)
        except Exception:
            prev = None

    benchmarks = [r for r in rows if r.get("asset_class") == "ETF"]
    payload = {
        "as_of": _now(),
        "model_version": "v3",
        "benchmark": uni.BENCHMARK,
        "universe_source": provenance,
        "universe": len(scored),
        "weights": model.WEIGHTS,
        "coverage": cov,
        "dispersion": round(model.dispersion(scored), 2),
        "price_failures": failures,
        "turnover": turnover(prev, scored),
        "all": scored,
        "benchmarks": benchmarks,
        "top": scored[:25],
    }
    with open(prev_path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {prev_path} ({len(scored)} scored, {len(benchmarks)} benchmarks)")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the conviction ledger.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of equities (ETFs always included)")
    ap.add_argument("--skip-macro", action="store_true")
    ap.add_argument("--skip-fundamentals", action="store_true",
                    help="prices only; useful for isolating a price-source problem")
    args = ap.parse_args()
    build(args.limit, args.skip_macro, args.skip_fundamentals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
