#!/usr/bin/env python3
"""Re-measure every claim in FINDINGS.md, from live keyless sources.

Run this before believing anything in that document. It reuses the production price
loader on purpose: the point of pillars 1, 3 and 4 on the equity side is that they
need no data the nightly job does not already have in memory.

    python3 research/probe.py [--tickers AAPL,MSFT,...] [--json out.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equity_monitor.sources import prices                      # noqa: E402
from research import structural_yield as sy                    # noqa: E402

DEFAULT = ["SPY", "V", "MA", "KO", "PEP", "HD", "LOW", "XOM", "CVX", "GS", "MS",
           "UPS", "FDX", "UNP", "CSX", "AAPL", "MSFT", "JPM"]

# Pairs a human would propose from shared economics, as the control group against
# the exhaustive search. If the search only beats these by data-mining, it has
# found nothing.
MOTIVATED = [("V", "MA"), ("KO", "PEP"), ("HD", "LOW"), ("XOM", "CVX"),
             ("GS", "MS"), ("UPS", "FDX"), ("UNP", "CSX"), ("AAPL", "MSFT")]

CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
FRED_VIX = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def fetch(url: str, headers: dict, tries: int = 4) -> str | None:
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as fh:
                return fh.read().decode("utf-8", "replace")
        except Exception as exc:                                # noqa: BLE001
            print(f"  retry {a + 1}: {exc}", file=sys.stderr)
            time.sleep(3 * (a + 1))
    return None


def load_bars(tickers: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for t in tickers:
        try:
            b = prices.load(t, asset_class="etf" if t == "SPY" else "stocks")
        except Exception as exc:                                # noqa: BLE001
            print(f"{t}: {exc}", file=sys.stderr)
            continue
        if b and len(b) > 300:
            out[t] = dict(zip(b.dates, b.close))
            print(f"{t:5s} {len(b):4d} bars via {b.source}", file=sys.stderr)
    return out


def aligned(bars, a, b, n=504):
    """Intersect on trading dates. Equal-length series on different dates produce a
    spread that measures the misalignment, not the relationship."""
    common = sorted(set(bars[a]) & set(bars[b]))[-n:]
    return [bars[a][d] for d in common], [bars[b][d] for d in common]


def pillar1(bars: dict) -> dict:
    control = {}
    for a, b in MOTIVATED:
        if a in bars and b in bars:
            ca, cb = aligned(bars, a, b)
            r = sy.engle_granger(ca, cb)
            if r:
                control[f"{a}/{b}"] = {
                    "adf": round(r["adf"], 2) if r["adf"] else None,
                    "beta": round(r["hedge_ratio"], 3),
                    "half_life_d": round(r["half_life"], 1) if r["half_life"] else None,
                    "z": round(r["z"], 2),
                    "passes_naive": bool(r["adf"] and r["adf"] < sy.EG_CRITICAL[0.05]),
                }
    found = []
    for a, b in itertools.combinations(sorted(bars), 2):
        if "SPY" in (a, b):
            continue
        ca, cb = aligned(bars, a, b)
        if len(ca) < 250:
            continue
        r = sy.engle_granger(ca, cb)
        if r and r["adf"] is not None:
            found.append((a, b, r))
    stats = [r["adf"] for _, _, r in found]
    cutoff = sy.benjamini_hochberg(stats, 0.05) if stats else None
    naive = [(a, b, r) for a, b, r in found if r["adf"] < sy.EG_CRITICAL[0.05]]
    survivors = [(a, b, r) for a, b, r in found if cutoff and sy.tradeable(r, cutoff)]
    return {
        "motivated_pairs": control,
        "tested": len(found),
        "expected_false_positives_5pct": round(0.05 * len(found), 1),
        "passed_naive_5pct": len(naive),
        "negative_hedge_ratio_among_passers":
            sum(1 for _, _, r in naive if r["hedge_ratio"] < 0),
        "fdr_cutoff": round(cutoff, 2) if cutoff else None,
        "tradeable_after_all_gates": [
            {"pair": f"{a}/{b}", "adf": round(r["adf"], 2),
             "beta": round(r["hedge_ratio"], 2),
             "half_life_d": round(r["half_life"], 1), "z": round(r["z"], 2)}
            for a, b, r in survivors],
    }


def atm_iv(sym: str, target_days: int = 25) -> dict | None:
    """Average IV of the contracts nearest the money in the nearest expiry beyond
    ``target_days``. Delayed 15 minutes, which is irrelevant to a nightly job."""
    txt = fetch(CBOE.format(sym=sym), {"User-Agent": "Mozilla/5.0"})
    if not txt:
        return None
    d = json.loads(txt).get("data", {})
    close = d.get("close")
    today = date.today()
    buckets: dict[int, list] = {}
    for o in d.get("options") or []:
        m = re.match(r"^[A-Z]+(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", o["option"])
        if not m or not o.get("iv") or close is None:
            continue
        yy, mm, dd, _cp, strike = m.groups()
        days = (date(2000 + int(yy), int(mm), int(dd)) - today).days
        if days >= target_days:
            buckets.setdefault(days, []).append((abs(int(strike) / 1000 - close), o["iv"]))
    if not buckets:
        return None
    days = min(buckets)
    near = sorted(buckets[days])[:8]
    return {"dte": days, "iv": sum(iv for _, iv in near) / len(near), "close": close}


def pillar2(bars: dict) -> dict:
    out = {}
    txt = fetch(FRED_VIX, {"User-Agent": "structural-yield-probe/1.0"})
    vix = {}
    if txt:
        for line in txt.splitlines()[1:]:
            p = line.split(",")
            if len(p) == 2 and p[1] not in (".", ""):
                vix[p[0]] = float(p[1])
    if vix and "SPY" in bars:
        dates = sorted(bars["SPY"])
        closes = [bars["SPY"][d] for d in dates]
        implied = [(d, v / 100.0) for d, v in vix.items() if d in bars["SPY"]]
        fwd = sy.forward_vrp(implied, closes, dates, window=21)
        rv21 = sy.realized_vol(closes, 21)
        last = max(vix)
        out["index"] = {
            "vix_now": vix[last], "vix_date": last, "spy_rv21_trailing": round(rv21, 4),
            "spread_now_vs_trailing": round(vix[last] / 100 - rv21, 4),
            "forward_vrp": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in fwd.items()} if fwd else None,
        }
    per = {}
    for t in ["SPY", "AAPL", "MSFT", "KO", "XOM", "JPM"]:
        if t not in bars:
            continue
        closes = [bars[t][d] for d in sorted(bars[t])]
        a = atm_iv(t)
        if not a:
            continue
        rv = sy.realized_vol(closes, 21)
        rv_ex = sy.realized_vol(closes, 21, drop_extremes=1)
        per[t] = {
            "atm_iv": round(a["iv"], 4), "dte": a["dte"],
            "rv21_trailing": round(rv, 4), "rv21_ex_largest_day": round(rv_ex, 4),
            "naive_vrp": round(a["iv"] - rv, 4),
            "event_adjusted_vrp": round(a["iv"] - rv_ex, 4),
        }
        time.sleep(1)
    out["per_name"] = per
    return out


def pillar3(bars: dict) -> dict:
    basket = [t for t in ["AAPL", "MSFT", "V", "MA", "KO", "PEP", "XOM", "CVX",
                          "HD", "UNP"] if t in bars]
    if len(basket) < 5:
        return {}
    common = sorted(set.intersection(*[set(bars[t]) for t in basket]))[-504:]
    series = [[bars[t][d] for d in common] for t in basket]
    w = [1 / len(basket)] * len(basket)
    cov = sy.cov_matrix(series)
    gamma = sy.excess_growth(cov, w)
    years = (len(common) - 1) / 252
    res = {"basket": basket, "days": len(common),
           "excess_growth_annual": round(gamma, 4),
           "theorem_predicts_ratio": round(math.exp(gamma * years), 4)}
    for every, label in ((1, "daily"), (21, "monthly"), (63, "quarterly")):
        bt = sy.rebalance_backtest(series, w, every=every)
        res[label] = {
            "rebalanced": round(bt["rebalanced"], 4),
            "buy_and_hold": round(bt["buy_and_hold"], 4),
            "geometric": round(bt["geometric"], 4),
            "measured_ratio_vs_geometric": round(bt["rebalanced"] / bt["geometric"], 4),
            "ratio_vs_buy_and_hold": round(bt["rebalanced"] / bt["buy_and_hold"], 4),
            "turnover": round(bt["turnover"], 2),
        }
    return res


def pillar4(bars: dict) -> dict:
    per, total, fwd_all = {}, 0, []
    for t in sorted(bars):
        closes = [bars[t][d] for d in sorted(bars[t])]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        hits, fwd5 = 0, []
        for i in range(60, len(rets) - 5):
            hist = rets[i - 60:i]
            mu = sum(hist) / 60
            sd = math.sqrt(sum((r - mu) ** 2 for r in hist) / 59)
            if sd and (rets[i] - mu) / sd <= -3:
                hits += 1
                fwd5.append(sum(rets[i + 1:i + 6]))
        total += hits
        fwd_all.extend(fwd5)
        per[t] = {"events_2y": hits,
                  "mean_fwd_5d_pct": round(100 * sum(fwd5) / len(fwd5), 2) if fwd5 else None}
    return {
        "per_name": per,
        "events_across_universe_2y": total,
        "events_per_name_per_year": round(total / max(1, len(bars)) / 2, 2),
        "pooled_mean_fwd_5d_pct": round(100 * sum(fwd_all) / len(fwd_all), 2) if fwd_all else None,
        "pooled_win_rate": round(sum(1 for f in fwd_all if f > 0) / len(fwd_all), 2) if fwd_all else None,
        "caveat": "2-8 events per name over 2y cannot size a per-name effect",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT))
    ap.add_argument("--json", default="")
    ap.add_argument("--skip-options", action="store_true",
                    help="skip CBOE chains (~1MB per name)")
    args = ap.parse_args()

    bars = load_bars([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    if not bars:
        print("no price data", file=sys.stderr)
        return 1

    out = {
        "as_of": max(max(v) for v in bars.values()),
        "pillar1_cointegration": pillar1(bars),
        "pillar2_vrp": {} if args.skip_options else pillar2(bars),
        "pillar3_harvest": pillar3(bars),
        "pillar4_overreaction": pillar4(bars),
    }
    text = json.dumps(out, indent=2)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
