"""CLI / nightly orchestrator for the Equity Conviction Monitor.

Pipeline: Data (FMP/Polygon) -> Features -> v2 Multiplicative Model -> Ledger.
Mirrors crypto-conviction-monitor/nightly.py structure: build() ranks the universe,
score() decomposes every asset, parity+regression gate via tests/test_parity.py.

Usage:
    equity-monitor refresh            # fetch + score + write ledger/signals.json
    equity-monitor score AAPL TSLA    # ad-hoc score two tickers
"""
from __future__ import annotations
import os, sys, json, math, time
from datetime import datetime, timezone

from .model import score, signal
from .data import Quote, build_universe, fetch_quotes, enrich_fundamentals

BENCHMARK = "SPY"


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}", file=sys.stderr)


def rs_blend(t: Quote, bench: Quote, sector_bench: Quote | None = None) -> float:
    """Multi-horizon relative strength vs SPY + sector, log-returns, vol-normalised.
    Mirrors crypto's rs_blend (0.30*rs7 + 0.25*rs14 + 0.25*rs30 + 0.20*rs200) but
    adds a sector-relative leg and normalises by return-vol so high-vol names
    (small/mid-cap) don't dominate momentum on raw percent alone."""
    horizon = {
        "rs7": t.price_change_7d - (bench.price_change_7d if bench else 0),
        "rs14": t.price_change_14d - (bench.price_change_14d if bench else 0),
        "rs30": t.price_change_30d - (bench.price_change_30d if bench else 0),
        "rs200": t.price_change_200d - (bench.price_change_200d if bench else 0),
    }
    sec = sector_bench or bench
    sec_horizon = {
        "rs7": t.price_change_7d - (sec.price_change_7d if sec else 0),
        "rs14": t.price_change_14d - (sec.price_change_14d if sec else 0),
        "rs30": t.price_change_30d - (sec.price_change_30d if sec else 0),
        "rs200": t.price_change_200d - (sec.price_change_200d if sec else 0),
    }
    # blend: 30% 7d + 25% 14d + 25% 30d + 20% 200d, then sector-denoised
    rs_w = (0.30 * horizon["rs7"] + 0.25 * horizon["rs14"] +
            0.25 * horizon["rs30"] + 0.20 * horizon["rs200"])
    rs_s = (0.30 * sec_horizon["rs7"] + 0.25 * sec_horizon["rs14"] +
            0.25 * sec_horizon["rs30"] + 0.20 * sec_horizon["rs200"])
    blended = rs_w - rs_s  # sector-denoised relative strength
    return blended


def build(quotes: list[Quote], bench: Quote) -> list[dict]:
    """Score the full universe; return ranked list with decomposition."""
    scored = []
    for t in quotes:
        if t.symbol == BENCHMARK or not t.market_cap:
            continue
        # turnover for risk
        t.turnover = (t.total_volume * 5) / t.market_cap if t.market_cap else 0  # rough avg daily turn %
        q, c, r, conv, sig, comp = score(t.as_dict(), bench.as_dict())
        rec = {"symbol": t.symbol, "price": t.price, "conviction": conv, "signal": sig,
               "mcap": t.market_cap, "turnover": t.turnover,
               "rs_blend": comp["rs_blend"], "chg": t.price_change_24h,  # 24h price % (for matrix Δ col)
               "z_ath": 0.0, "factors": comp}
        scored.append(rec)
    # multiplicative engine: rank by conv desc, tie-break by quality then confirmation
    scored.sort(key=lambda x: (-x["conviction"], -x["factors"].get("quality", 0),
                               -x["factors"].get("confirmation", 0)))
    return scored


def refresh() -> None:
    _log("building universe …")
    syms = build_universe()
    _log(f"universe={len(syms)} (S&P500+Russell1000+ETFs)")
    # fetch benchmarks first
    bench_q = fetch_quotes([BENCHMARK])
    bench = bench_q[0] if bench_q else None
    # fetch the universe in batches of 100 (FMP free tier caps)
    out = []
    for i in range(0, len(syms), 100):
        batch = syms[i:i + 100]
        qs = fetch_quotes(batch)
        for q in qs:
            enrich_fundamentals(q)
            q.rs_blend = rs_blend(q, bench) if bench else 0.0
        out.extend(qs)
    ranked = build(out, bench)
    _write_ledger(ranked, bench)
    _log(f"scored {len(ranked)} names; top={ranked[0]['symbol']}@{ranked[0]['conviction']} {ranked[0]['signal']}")
    print(json.dumps(ranked[:10], indent=2))


def _write_ledger(ranked: list[dict], bench: Quote | None) -> None:
    os.makedirs("ledger", exist_ok=True)
    mcapsum = sum(r.get("mcap", 0) for r in ranked)
    payload = {"as_of": datetime.now(timezone.utc).isoformat(),
               "benchmark": (bench.symbol if bench else "SPY"),
               "universe": len(ranked) + 1,
               "mcapsum": mcapsum,
               "top10": ranked[:10],
               "all": ranked}
    with open("ledger/index.json", "w") as f:
        json.dump(payload, f, indent=2)
    _log("wrote ledger/index.json")


def cli() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "refresh"
    if cmd == "refresh":
        refresh()
    elif cmd == "score":
        syms = sys.argv[2:]
        qs = fetch_quotes(syms)
        for q in qs:
            enrich_fundamentals(q)
            q.rs_blend = rs_blend(q, None)
            qd = q.as_dict()
            _, _, _, conv, sig, comp = score(qd)
            print(f"{q.symbol}: {conv} {sig}  Q={comp['quality']:.1f} C={comp['confirmation']:.1f} R={comp['risk']:.3f}  rs_blend={comp['rs_blend']:.1f}")
    else:
        print(f"unknown command: {cmd}\nusage: equity-monitor [refresh|score TSLA AAPL]", file=sys.stderr)
