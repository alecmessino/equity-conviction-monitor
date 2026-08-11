"""Reconstruct the board as of an earlier session, and diff today against it.

The snapshot series accumulates one observation per trading day and cannot be
back-filled, so for the first weeks of its life there is nothing to compare against.
This module answers the questions the diff and the churn diagnostic are built to answer
— what changed, and was it the data or the model — using price history the project
already has on disk, without touching the recorded series.

**What is real and what is not.** Every price-derived input is genuinely reconstructed:
relative strength, trend, dollar liquidity and realised volatility are recomputed from
the actual OHLCV bars, truncated to the cutoff session, and ranked cross-sectionally
among the names that existed then. That is a true board for that date on the
confirmation and risk pillars.

The fundamentals are **today's**. Filings are not stored as-of-date, so a reconstruction
reads the current values back onto an earlier board. That is look-ahead bias on the
quality pillar. Over a few weeks it is small — a filer that has not reported since the
cutoff has identical values either way, and one that has reported is a handful of names
— and over a quarter it is not small at all. It is stated on every output this module
produces rather than buried here.

**Why it may never write to ``ledger/snapshots/``.** A reconstruction carries that
look-ahead, and an Information Coefficient computed over a series containing even a few
reconstructed nights would be measuring the leak rather than the model. The output path
is checked and the run refuses if it is aimed at the recorded history. That check is the
single most important line in this file.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equity_monitor import (churn, features, model, monitor, snapshots,
                            universe as uni, watchlist)
from equity_monitor.sources import prices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "ledger")

# A series needs this many bars before its 200-day trend and 12-month excess return are
# anything but noise. Below it the name is dropped rather than scored on a short window,
# which is the same rule the nightly applies.
MIN_BARS = 60

# Fields lifted from the published ledger rather than recomputed. Everything here is
# filing-derived; every price-derived input is deliberately absent so the reconstruction
# recomputes it. Keeping the two sets disjoint is what makes the look-ahead bounded and
# nameable rather than diffuse.
FUNDAMENTAL_FIELDS = tuple(
    [m for _, m, _ in model.RANK_SPEC
     if m not in ("rs_blend", "trend", "adv_usd", "vol_1y")]
    + ["earnings_yield", "ebitda_yield", "market_cap", "pe", "shares"]
)

DISCLOSURE = (
    "Reconstructed board. Price inputs are recomputed from real OHLCV truncated to the "
    "cutoff; fundamentals are today's published values, which is look-ahead bias on the "
    "quality pillar. Not part of the recorded history and not usable for return analysis."
)


class RefusedError(RuntimeError):
    """Raised when a reconstruction is aimed at the recorded snapshot series."""


def _assert_not_the_real_history(out_dir: str) -> None:
    """Refuse to write a reconstruction into the recorded series.

    An Information Coefficient computed over a series containing reconstructed nights
    measures the look-ahead, not the model, and the contamination is undetectable after
    the fact because a reconstructed snapshot is structurally identical to a real one.
    """
    real = os.path.realpath(os.path.join(LEDGER, "snapshots"))
    target = os.path.realpath(os.path.join(out_dir, "snapshots"))
    if target == real or target.startswith(real + os.sep):
        raise RefusedError(
            f"refusing to write a reconstructed board into {real}. Reconstructions carry "
            "look-ahead on the quality pillar; mixing them into the recorded series would "
            "corrupt every return study run against it, undetectably. Choose an output "
            "directory outside ledger/."
        )


def load_bars(history_dir: str) -> dict[str, dict]:
    """Every cached OHLCV series, as raw dicts."""
    out: dict[str, dict] = {}
    for path in glob.glob(os.path.join(history_dir, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if d.get("symbol") and d.get("dates"):
            out[d["symbol"]] = d
    return out


def sessions(bars: dict[str, dict], benchmark: str = uni.BENCHMARK) -> list[str]:
    """The trading calendar, taken from the benchmark.

    One calendar for the whole universe, so every name truncates to the same session. A
    per-symbol "last N bars" rule would silently compare names as of different dates
    whenever one of them did not trade.
    """
    node = bars.get(benchmark)
    if not node:
        raise RuntimeError(
            f"benchmark {benchmark} has no cached history, so there is no calendar to "
            "truncate against and no reference for relative strength"
        )
    return list(node["dates"])


def truncate(node: dict, cutoff: str) -> prices.Bars | None:
    """One series, cut to end at ``cutoff``. None if too little history remains."""
    keep = [i for i, stamp in enumerate(node["dates"]) if stamp <= cutoff]
    if len(keep) < MIN_BARS:
        return None
    lo, hi = keep[0], keep[-1] + 1
    return prices.Bars(
        symbol=node["symbol"], dates=node["dates"][lo:hi],
        open=node["open"][lo:hi], high=node["high"][lo:hi],
        low=node["low"][lo:hi], close=node["close"][lo:hi],
        volume=node["volume"][lo:hi],
    )


def board(cutoff: str, bars: dict[str, dict], published: dict[str, dict],
          members: list) -> list[dict]:
    """Score the universe as of ``cutoff``.

    Price features come from truncated bars and are re-ranked among the names that had
    history then, which matters: a percentile is a statement about a peer group, and
    ranking a 2026 board against a peer group that includes names listed later would be
    a subtler leak than the fundamental one.
    """
    truncated = {sym: b for sym, node in bars.items()
                 if (b := truncate(node, cutoff)) is not None}
    if uni.BENCHMARK not in truncated:
        raise RuntimeError(f"benchmark {uni.BENCHMARK} has too little history before "
                           f"{cutoff} to measure relative strength against")

    rows = features.build(members, truncated, today=date.fromisoformat(cutoff),
                          fundamentals={})

    for r in rows:
        src = published.get(r["symbol"])
        if not src:
            continue
        for key in FUNDAMENTAL_FIELDS:
            if src.get(key) is not None:
                r[key] = src[key]

    model.score_rows(rows)
    scored = [r for r in rows if r.get("conviction") is not None]
    scored.sort(key=lambda r: r["conviction"], reverse=True)

    # Mirrors the nightly so the reconstructed rows are shaped identically to recorded
    # ones — the diff and churn modules read both through the same decoder.
    from equity_monitor.nightly import assign_weights, round_row
    assign_weights(scored)
    for r in rows:
        round_row(r)
    return scored


def run(back: int, out_dir: str, ledger_dir: str = LEDGER) -> dict:
    """Reconstruct the board ``back`` sessions ago and diff the current one against it."""
    _assert_not_the_real_history(out_dir)

    with open(os.path.join(ledger_dir, "index.json")) as fh:
        payload = json.load(fh)
    published = {r["symbol"]: r for r in (payload.get("all") or []) if r.get("symbol")}

    bars = load_bars(os.path.join(ledger_dir, "history"))
    if not bars:
        raise RuntimeError(
            f"no cached price history in {ledger_dir}/history — a reconstruction is "
            "computed from the bars on disk and cannot be done without them"
        )
    calendar = sessions(bars)
    if back < 1 or back >= len(calendar):
        raise ValueError(f"--back must be between 1 and {len(calendar) - 1}; "
                         f"the cached calendar runs {calendar[0]} to {calendar[-1]}")
    cutoff = calendar[-1 - back]

    members, _ = uni.load(None)
    scored = board(cutoff, bars, published, members)

    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
    # cutoff comes from the benchmark's trading calendar, so it *is* a real session and
    # can be recorded as one. `coverage` is 1.0 by construction here: sessions() already
    # truncates every name to the same calendar, which is the whole reason a
    # reconstruction can be dated at all.
    snapshots.write(scored, out_dir, on=cutoff, as_of=cutoff + "T23:00:00Z",
                    session={"session": cutoff, "coverage": 1.0,
                             "symbols": len(scored), "spread": {},
                             "source": "reconstructed from the benchmark calendar"})

    # The current recorded snapshot is copied in as the second point, so the diff runs
    # against the real board rather than a second reconstruction.
    latest = snapshots.available(ledger_dir)
    if not latest:
        raise RuntimeError("no recorded snapshot to diff against — run the nightly first")
    today_stamp = latest[-1]
    shutil.copy(snapshots.snapshot_path(ledger_dir, today_stamp),
                snapshots.snapshot_path(out_dir, today_stamp))

    prior = snapshots.read(snapshots.snapshot_path(out_dir, cutoff))
    current = snapshots.read(snapshots.snapshot_path(out_dir, today_stamp))

    attribution = {}
    for sym in sorted(set(prior["rows"]) & set(current["rows"])):
        a = snapshots.attribute(prior["rows"][sym], current["rows"][sym])
        if a:
            attribution[sym] = a
    meta = {sym: {"name": r.get("name"), "sector": r.get("sector")}
            for sym, r in published.items()}

    diff = watchlist.diff(prior, current, attribution, meta)
    diagnosis = churn.diagnose(prior, current)
    stability = monitor.stability(prior, current)

    for node in (diff, diagnosis):
        if node is not None:
            node["reconstructed"] = True
            node["disclosure"] = DISCLOSURE

    return {
        "cutoff": cutoff, "against": today_stamp, "sessions_back": back,
        "names": len(scored), "out_dir": out_dir,
        "stability": stability, "churn": diagnosis, "watchlist": diff,
        "reconstructed": True, "disclosure": DISCLOSURE,
    }


def _write_site(out_dir: str, result: dict, ledger_dir: str = LEDGER) -> None:
    """Assemble a browsable copy of the terminal pointed at the reconstruction."""
    web = os.path.join(out_dir, "web")
    os.makedirs(web, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "web", "terminal.html"),
                os.path.join(web, "terminal.html"))
    for name in ("index.json", "macro.json", "history.json", "trends.json"):
        src = os.path.join(ledger_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))

    with open(os.path.join(out_dir, "index.json")) as fh:
        payload = json.load(fh)
    macro = None
    macro_path = os.path.join(ledger_dir, "macro.json")
    if os.path.exists(macro_path):
        with open(macro_path) as fh:
            macro = json.load(fh)

    report = monitor.build(out_dir, payload, macro)
    report["churn"] = result["churn"]
    report["reconstructed"] = True
    report["disclosure"] = DISCLOSURE
    with open(os.path.join(out_dir, "monitor.json"), "w") as fh:
        json.dump(report, fh, separators=(",", ":"))
    with open(os.path.join(out_dir, "watchlist.json"), "w") as fh:
        json.dump(result["watchlist"], fh, separators=(",", ":"))


def _report(result: dict) -> None:
    s, c, d = result["stability"], result["churn"], result["watchlist"]
    print(f"\ncutoff {result['cutoff']} → {result['against']} "
          f"({result['sessions_back']} sessions, {result['names']} names)")
    print(f"  {DISCLOSURE}\n")

    if s:
        print("stability")
        print(f"  rank correlation   {s['rank_correlation']}")
        print(f"  mean |Δ|           {s['mean_abs_move']}  median {s['median_abs_move']}  "
              f"max {s['max_abs_move']}")
        print(f"  tier changes       {s['tier_changes']}  ({s['unchanged']} scored identically)")
    if c:
        print("\nchurn")
        print(f"  assessment         {c['assessment']}")
        print(f"  input movement     {c['input_movement']}   "
              f"standing movement {c['output_movement']}   ratio {c['amplification']}x")
        lead = c.get("largest_contributor")
        if lead:
            print(f"  led by             {lead['factor']} ({lead['share']:.0%})")
        b = c["boundary"]
        print(f"  boundary           {b['marginal']}/{b['tier_changes']} tier changes were marginal")
    if d:
        counts = d["counts"]
        print("\nwatchlist")
        print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        for section, label in (("upgrades", "UP"), ("downgrades", "DOWN")):
            for row in d[section][:8]:
                drv = row["driver"]
                print(f"  {label:4} {row['symbol']:6} {(row['name'] or '')[:24]:24} "
                      f"{row['from_tier']:6}→{row['to_tier']:6} "
                      f"{row['from']:3}→{row['to']:3} ({row['delta']:+3d})  "
                      f"{drv['factor'] if drv else '—'}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--back", type=int, default=5,
                    help="trading sessions to reconstruct backwards (default 5)")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "rebuild"),
                    help="output directory; may not be inside ledger/snapshots")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--json", action="store_true", help="emit the full result as JSON")
    ap.add_argument("--site", action="store_true",
                    help="also assemble a browsable terminal pointed at the reconstruction")
    args = ap.parse_args()

    try:
        result = run(args.back, args.out, args.ledger)
    except RefusedError as exc:
        print(f"REFUSED  {exc}")
        return 2
    except Exception as exc:
        print(f"FAILED   {exc}")
        return 1

    if args.site:
        _write_site(args.out, result, args.ledger)
        print(f"site assembled at {args.out}/web/terminal.html")
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
