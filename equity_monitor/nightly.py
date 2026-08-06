"""Nightly builder — Equity Conviction Monitor.

Pulls features for the universe (Yahoo keyless prices + Alpha Vantage fundamentals),
enriches with RS, scores via model.score(), and writes ledger/index.json.

If ALPHAVANTAGE_API_KEY is absent, prices still pull keylessly from Yahoo and the
terminal renders live prices with neutral Quality (never fabricated). If the pull
produces zero rows (network/API down), it falls back to the committed fixture so
the terminal never goes empty.
"""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equity_monitor.data as data
import equity_monitor.rs as rs
from equity_monitor import model
from equity_monitor.model import score, signal

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ledger")

BENCH = {"symbol": "SPY"}


def build(allow_fixture_fallback=True):
    av_set = bool(data.AV_KEY)
    print(f"Alpha Vantage key: {'SET' if av_set else 'unset (neutral fundamentals, Yahoo prices only)'}")

    symbols = data.universe()
    feats, skipped = [], []
    for sym in symbols:
        try:
            f = data.build_features(sym)
            if f:
                feats.append(f)
            else:
                skipped.append(sym)
        except Exception as e:
            skipped.append(f"{sym}:{e}")
        time.sleep(0.05)  # gentle on rate limits

    if not feats:
        # live pull produced nothing (network/API down) — never ship an empty terminal
        print("live pull produced 0 rows — falling back to committed fixture.")
        if allow_fixture_fallback:
            _fallback_to_committed()
        else:
            write_ledger([], 0.0)
        return

    feats = rs.enrich(feats)
    rows = []
    hist_dir = os.path.join(LEDGER, "history")
    os.makedirs(hist_dir, exist_ok=True)
    for f in feats:
        q, c, r, conv, sig, comp = score(f, BENCH)
        sym = f["symbol"]
        # persist trailing closes for sparklines (best-effort; keyless Yahoo)
        try:
            closes = data.history(sym, 30)
            if closes:
                with open(os.path.join(hist_dir, f"{sym}.json"), "w") as hf:
                    json.dump({"symbol": sym, "closes": closes}, hf)
        except Exception as e:
            print(f"  history {sym} skipped: {e}")
        rows.append({
            "symbol": sym,
            "price": round(f["price"], 2),
            "chg": round(f.get("chg24", 0.0), 2),
            "turnover": round(f.get("turnover", 0.0), 5),
            "adv": int(f.get("adv", 0)),
            "mcap": int(f.get("market_cap", 0)),
            "z_ath": None,
            "conviction": conv,
            "signal": sig,
            "sector": f.get("sector", ""),
            "beta": f.get("beta"),  # None when unavailable (no AV key) -> UI shows "—"
            "dte": f.get("dte"),  # days to next earnings (None = n/a, e.g. ETFs)
            "factors": {
                "quality": comp["quality"],
                "confirmation": comp["confirmation"],
                "risk": comp["risk"],
                "rs_blend": comp["rs_blend"],
                "liquidity_fit": comp["liquidity_fit"],
                "val_zscore": comp["val_zscore"],
                "short_days": comp.get("short_days", 0.0),
                # raw fundamentals in FRACTION units (matching model.py inputs so the
                # browser JS port reproduces score() exactly — see frontend/backend parity)
                "roic": round(f.get("roic", 0.0) or 0.0, 4),
                "fcf_yield": round(f.get("fcf_yield", 0.0) or 0.0, 4),
                "gross_margin": round(f.get("gross_margin", 0.0) or 0.0, 4),
                "debt_ebitda": round(f.get("debt_ebitda", 5.0) or 5.0, 2),
                "earnings_stability": round(f.get("earnings_stability", 0.5) or 0.5, 2),
            },
        })
    rows.sort(key=lambda x: x["conviction"], reverse=True)
    # score-proportional weights (top names get larger weight; normalized to 100)
    total_score = sum(max(r["conviction"], 1) for r in rows)
    for r in rows:
        r["weight"] = round(max(r["conviction"], 1) / total_score * 100.0, 2)
    mcapsum = sum(r["mcap"] for r in rows)
    write_ledger(rows, mcapsum, skipped, live=True)
    print(f"built {len(rows)} rows from live data, skipped {len(skipped)}: {skipped[:5]}")


def _fallback_to_committed():
    """Copy the committed ledger/index.json as-is so the terminal never goes empty."""
    src = os.path.join(LEDGER, "index.json")
    if os.path.exists(src):
        with open(src) as fh:
            payload = json.load(fh)
        payload["as_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload["live"] = False
        with open(src, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"fallback: served committed fixture ({len(payload.get('all', []))} rows)")
    else:
        write_ledger([], 0.0)


def compute_sector_deltas(prev_payload, rows):
    """Score-weighted sector % now vs previous run. Additive metadata only."""
    def sector_weights(payload):
        by = {}
        for r in (payload or {}).get("all", []):
            w = r.get("weight", 0) or 0
            sec = (r.get("factors") or {}).get("sector") or r.get("sector") or "Other"
            by[sec] = by.get(sec, 0) + w
        tot = sum(by.values()) or 1.0
        return {k: v / tot * 100.0 for k, v in by.items()}
    cur = sector_weights({"all": rows})
    prev = sector_weights(prev_payload)
    out = {}
    for sec in set(cur) | set(prev):
        out[sec] = round(cur.get(sec, 0.0) - prev.get(sec, 0.0), 2)
    return out


def write_ledger(rows, mcapsum, skipped=None, live=False):
    import glob
    os.makedirs(LEDGER, exist_ok=True)
    # archive previous run for drift comparison (rolling T-1)
    prev_path = os.path.join(LEDGER, "prev_index.json")
    prev_payload = None
    if os.path.exists(prev_path):
        try:
            with open(prev_path) as fh:
                prev_payload = json.load(fh)
        except Exception:
            prev_payload = None
    # sector deltas vs previous run
    sector_deltas = compute_sector_deltas(prev_payload, rows) if prev_payload else {}
    payload = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": "SPY",
        "live": live,
        "hist_base": "history/",   # sparkline closes live at {hist_base}{SYMBOL}.json
        "sector_deltas": sector_deltas,  # additive metadata: today% - prev% per sector
        "universe": len(rows),
        "mcapsum": float(mcapsum),
        "skipped": skipped or [],
        "all": rows,
        "top10": rows[:10],
    }
    with open(os.path.join(LEDGER, "index.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    # archive this run as previous for next drift comparison
    try:
        with open(prev_path, "w") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    build()
