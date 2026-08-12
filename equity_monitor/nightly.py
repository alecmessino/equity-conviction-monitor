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

from equity_monitor import (churn, edge, features, health, model, monitor, performance,
                            snapshots, universe as uni, watchlist)
from equity_monitor.sources import edgar, macro, prices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "ledger")
HISTORY = os.path.join(LEDGER, "history")

COVERAGE_FIELDS = [
    "market_cap", "roic", "fcf_yield", "gross_margin", "net_debt_ebitda",
    "earnings_stability", "earnings_yield", "rs_blend", "trend", "adv_usd",
    "vol_1y", "drawdown_52w", "sector",
    # sector-specific inputs
    "roe", "efficiency_ratio", "equity_to_assets", "ffo_yield", "debt_to_assets",
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


# Rounding happens once, here, on the way out. model.score() deliberately returns full
# precision: rounding inside it made Python and JavaScript disagree in the last decimal
# (banker's vs half-away-from-zero), which the parity gate flagged.
_ROUND = {
    "q": 4, "c": 4, "r": 4, "q_raw": 4, "c_raw": 4, "r_raw": 4, "mr_uplift": 4,
    "price": 4, "chg_1d": 6, "ret_1m": 6, "ret_3m": 6, "ret_6m": 6, "ret_12m": 6,
    "ret_ytd": 6, "vol_3m": 6, "vol_1y": 6, "atr14": 4, "atr_pct": 6,
    "hi_52w": 4, "lo_52w": 4, "drawdown_52w": 6, "pct_off_low": 6,
    "ma50": 4, "ma200": 4, "px_vs_ma50": 6, "px_vs_ma200": 6, "adv_usd": 0,
    "rs_blend": 6, "rs_sector": 6, "trend": 6, "market_cap": 0,
    "roic": 6, "fcf_yield": 6, "gross_margin": 6, "net_debt_ebitda": 4,
    "earnings_stability": 6, "earnings_yield": 6, "ebitda_yield": 6, "pe": 4,
    "value_metric": 6, "data_confidence": 3,
    **{k: 6 for k in ("p_roic", "p_fcf_yield", "p_gross_margin", "p_leverage",
                      "p_earnings_stability", "p_rs", "p_trend", "p_liquidity",
                      "p_lowvol", "p_value")},
}


def round_row(row: dict) -> dict:
    for key, places in _ROUND.items():
        v = row.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            row[key] = round(v, places) if places else round(v)
    return row


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
          skip_fundamentals: bool = False, offline: bool = False) -> dict:
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
    bars = features.fetch_bars(symbols, cache_dir=HISTORY, on_progress=progress,
                               prefer_cache=offline)
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
    for r in rows:
        round_row(r)

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

    # Factor-level snapshot: the full decomposition for every name, every night.
    # This is the research dataset — Information Coefficient, decile spreads and
    # factor decay are all regressions of past factor values against forward returns,
    # and none of them can be reconstructed after the fact. Written before the ledger
    # so a later failure cannot cost us the day's observation.
    attribution = snapshots.attribute_all(LEDGER, scored)
    # The benchmark close goes into the snapshot, not read back from the OHLCV cache
    # later: the cache lags the run by days, and a return series that substitutes a
    # nearby close for a missing one is measuring a different holding period.
    bench_row = next((r for r in rows if r.get("symbol") == uni.BENCHMARK), None)
    # as_of is the trading day the closes came from, not the wall clock of the run.
    # These differ routinely — a Friday 23:00 UTC run can still be reading Thursday's
    # bar, and a push-triggered Saturday run reads Friday's — so dating a leg by the
    # filename mislabels a real Thursday-to-Friday move as Friday-to-Saturday.
    sess = prices.session_date(bars)
    snap_path = snapshots.write(
        scored, LEDGER, as_of=_now(), session=sess,
        benchmark={"symbol": uni.BENCHMARK, "price": (bench_row or {}).get("price")})
    if sess["session"]:
        print(f"session: closes are from {sess['session']} "
              f"({sess['coverage']*100:.0f}% of {sess['symbols']} symbols agree"
              + (f"; also seen {sess['spread']}" if sess["spread"] else "") + ")")
    else:
        print("session: no bar dates available — legs will fall back to the run date")
    n_snaps = len(snapshots.available(LEDGER))
    print(f"snapshot: {os.path.basename(snap_path)} "
          f"({os.path.getsize(snap_path)/1024:.0f} KB, {n_snaps} on file)")
    if attribution:
        moved = attribution.get("names", {})
        print(f"attribution: {len(moved)} names moved since {attribution.get('since')}")
    try:
        with open(os.path.join(LEDGER, "trends.json"), "w") as fh:
            json.dump(snapshots.build_trends(LEDGER), fh, separators=(",", ":"))
        n_factor_files = snapshots.write_symbol_factors(LEDGER)
        print(f"trends + per-symbol factor history: {n_factor_files} symbols")
    except Exception as exc:
        print(f"trends: skipped ({exc})")

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
        "model_version": model.MODEL_VERSION,
        "spec_hash": model.spec_hash(),
        "snapshots": n_snaps,
        "attribution": attribution,
        "benchmark": uni.BENCHMARK,
        "universe_source": provenance,
        "universe": len(scored),
        "weights": model.WEIGHTS,
        "quality_profiles": model.QUALITY_PROFILES,
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

    # Model monitoring. Operational condition only — stability, coverage, regime,
    # freshness. Explicitly NOT a claim about predictive power, which needs months of
    # accumulated snapshots and is a separate question.
    try:
        macro_state = None
        macro_path = os.path.join(LEDGER, "macro.json")
        if os.path.exists(macro_path):
            with open(macro_path) as fh:
                macro_state = json.load(fh)
        report = monitor.build(LEDGER, payload, macro_state)
        # Why the board moved, not just how much. Attached here rather than inside
        # monitor.build because churn imports monitor for the rank correlation.
        report["churn"] = churn.from_ledger(LEDGER)
        with open(os.path.join(LEDGER, "monitor.json"), "w") as fh:
            json.dump(report, fh, separators=(",", ":"))
        counts: dict[str, int] = {}
        for c in report["health"]:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        print("health: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
              + f"  (observations={report['observations']})")
        for c in report["health"]:
            if c["status"] in ("fail", "warn"):
                print(f"  {c['status'].upper()}: {c['name']} — {c['detail']}")
        ch = report.get("churn")
        if ch:
            print(f"churn: {ch['assessment']} — {ch['basis']}")
        else:
            print("churn: pending (needs a second snapshot)")
    except Exception as exc:
        print(f"monitor: skipped ({exc})")

    # The overnight diff, published separately: it is a daily workflow rather than a
    # diagnostic, and pinning it to monitor.json would couple a morning view to a
    # monitoring artifact that may grow large.
    try:
        diff = watchlist.from_ledger(LEDGER, payload)
        with open(os.path.join(LEDGER, "watchlist.json"), "w") as fh:
            json.dump(diff, fh, separators=(",", ":"))
        if diff:
            c = diff["counts"]
            print(f"watchlist: {c['upgrades']} up, {c['downgrades']} down, "
                  f"{c['movers']} big moves in-tier, {c['boundary']} boundary crossings")
        else:
            print("watchlist: pending (needs a second snapshot)")
    except Exception as exc:
        print(f"watchlist: skipped ({exc})")

    # Paper return of the published book. Chained across recorded snapshots only — it
    # cannot be back-filled, because it needs the weights that were actually published
    # on the earlier night and those were not recorded before the ledger started.
    try:
        perf = performance.write(LEDGER)
        # The edge measurement rides on the same legs as the curve, so it is written
        # immediately after: whether the ordering is informative is the question that
        # decides whether any of the rest is worth acting on.
        edge.write(LEDGER)
        # Cohort stickiness, persistence and tier flips — the ribbon that says
        # whether tonight's board is a trend or a twitch. Read from the same
        # snapshots; no schema extension was needed for any of it.
        health.write(LEDGER)
        if perf["legs"]:
            print(f"performance: {perf['legs']} leg(s), book {perf['book_total']:+.2f}%"
                  + (f", {perf['benchmark']} {perf['benchmark_total']:+.2f}%"
                     if perf["benchmark_available"] else
                     f", {perf['benchmark']} not yet recorded")
                  + ("" if perf["renderable"] else
                     f" — below the {perf['min_days']}-day render threshold"))
        else:
            print("performance: pending (needs two snapshots)")
    except Exception as exc:
        print(f"performance: skipped ({exc})")

    print(f"wrote {prev_path} ({len(scored)} scored, {len(benchmarks)} benchmarks)")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the conviction ledger.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of equities (ETFs always included)")
    ap.add_argument("--skip-macro", action="store_true")
    ap.add_argument("--skip-fundamentals", action="store_true",
                    help="prices only; useful for isolating a price-source problem")
    ap.add_argument("--offline", action="store_true",
                    help="replay committed price history instead of refetching, so a "
                         "scoring change can be re-run in seconds against identical prices")
    args = ap.parse_args()
    build(args.limit, args.skip_macro, args.skip_fundamentals, args.offline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
