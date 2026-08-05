"""Nightly builder — Equity Conviction Monitor.

Pulls features for the universe, enriches with RS, scores via model.score(),
and writes ledger/index.json. Requires FMP_API_KEY (env). If the key is absent
or a symbol fails, that symbol is skipped (never fabricated). The terminal
renders whatever the ledger contains — so a partial pull still shows real data.
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

def build(allow_fixture_fallback=True):
    if not data.FMP_KEY:
        print("FMP_API_KEY not set — falling back to committed fixture ledger.")
        if allow_fixture_fallback:
            _fallback_to_committed()
        else:
            write_ledger([], 0.0)
        return

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

    feats = rs.enrich(feats)
    rows = []
    for f in feats:
        q, c, r, conv, sig, comp = score(f)
        rows.append({
            "symbol": f["symbol"],
            "price": round(f["price"], 2),
            "chg": round(f.get("chg24", 0.0), 2),
            "turnover": round(f.get("turnover", 0.0), 5),
            "adv": int(f.get("adv", 0)),
            "mcap": int(f.get("market_cap", 0)),
            "z_ath": None,
            "conviction": conv,
            "signal": sig,
            "sector": f.get("sector", ""),
            "beta": f.get("beta", 1.0),
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
    if not rows:
        # live pull produced nothing (bad key / API down) — never ship an empty terminal
        print("live pull produced 0 rows — falling back to committed fixture.")
        _fallback_to_committed()
        return
    write_ledger(rows, mcapsum, skipped)
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


def write_ledger(rows, mcapsum, skipped=None):
    os.makedirs(LEDGER, exist_ok=True)
    payload = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": "SPY",
        "universe": len(rows),
        "mcapsum": float(mcapsum),
        "skipped": skipped or [],
        "all": rows,
        "top10": rows[:10],
    }
    with open(os.path.join(LEDGER, "index.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

if __name__ == "__main__":
    build()
