#!/usr/bin/env python3
"""Generate a sample ledger/index.json for offline dev (mirrors nightly build output schema).
NOT a production data source — exercises the terminal render without API calls."""
import json, os
from equity_monitor import score

FIXTURE = {
    "AAPL": dict(symbol="AAPL", market_cap=3.4e12, price=220.0, price_change_24h=1.2,
                 price_change_7d=4.0, price_change_14d=2.0, price_change_30d=9.0, price_change_200d=-6.0,
                 roic=0.30, fcf_yield=0.08, gross_margin=0.42, debt_ebitda=1.8,
                 earnings_stability=0.85, val_zscore=-0.5, short_days=3.0,
                 drawdown_52w=0.14, total_volume=4.5e9),
    "MSFT": dict(symbol="MSFT", market_cap=2.8e12, price=480.0, price_change_24h=0.8,
                 price_change_7d=3.5, price_change_14d=1.0, price_change_30d=8.0, price_change_200d=-2.0,
                 roic=0.28, fcf_yield=0.07, gross_margin=0.68, debt_ebitda=0.4,
                 earnings_stability=0.90, val_zscore=0.3, short_days=2.0,
                 drawdown_52w=0.12, total_volume=2.8e9),
    "NVDA": dict(symbol="NVDA", market_cap=2.7e12, price=1100.0, price_change_24h=3.1,
                 price_change_7d=8.0, price_change_14d=22.0, price_change_30d=35.0, price_change_200d=45.0,
                 roic=0.28, fcf_yield=0.01, gross_margin=0.72, debt_ebitda=1.5,
                 earnings_stability=0.60, val_zscore=2.4, short_days=2.5,
                 drawdown_52w=0.05, total_volume=3.5e9),
    "BRK.B": dict(symbol="BRK.B", market_cap=7.5e11, price=520.0, price_change_24h=0.3,
                  price_change_7d=2.0, price_change_14d=1.5, price_change_30d=5.0, price_change_200d=-4.0,
                  roic=0.09, fcf_yield=0.035, gross_margin=0.50, debt_ebitda=0.2,
                  earnings_stability=0.95, val_zscore=0.8, short_days=1.0,
                  drawdown_52w=0.02, total_volume=3.0e9),
    "KO": dict(symbol="KO", market_cap=7.2e10, price=62.0, price_change_24h=-0.5,
               price_change_7d=-2.0, price_change_14d=-1.0, price_change_30d=1.0, price_change_200d=-12.0,
               roic=0.12, fcf_yield=0.07, gross_margin=0.62, debt_ebitda=2.8,
               earnings_stability=0.98, val_zscore=-1.2, short_days=9.0,
               drawdown_52w=0.08, total_volume=1.2e9),
    "IWM": dict(symbol="IWM", market_cap=6.1e11, price=210.0, price_change_24h=0.6,
                price_change_7d=3.0, price_change_14d=2.5, price_change_30d=6.0, price_change_200d=-10.0,
                roic=0.02, fcf_yield=0.01, gross_margin=0.10, debt_ebitda=0.0,
                earnings_stability=0.50, val_zscore=0.5, short_days=3.0,
                drawdown_52w=0.11, total_volume=4.0e9),
}
SPY = dict(symbol="SPY", market_cap=2.1e12, price_change_7d=2.5, price_change_14d=1.0,
           price_change_30d=4.0, price_change_200d=-3.0, price_change_24h=0.4)


def blend(t, b):
    return (0.30 * (t["price_change_7d"] - b["price_change_7d"]) +
            0.25 * (t["price_change_14d"] - b["price_change_14d"]) +
            0.25 * (t["price_change_30d"] - b["price_change_30d"]) +
            0.20 * (t["price_change_200d"] - b["price_change_200d"]))


def main():
    rows, mcsum = [], 0
    for s, t in FIXTURE.items():
        t["rs_blend"] = blend(t, SPY)
        t["turnover"] = t["total_volume"] / t["market_cap"]
        _, _, _, conv, sig, comp = score(t, SPY)
        mcsum += t["market_cap"]
        rows.append({"symbol": s, "price": t["price"], "conviction": conv, "signal": sig,
                     "mcap": t["market_cap"], "turnover": t["turnover"],
                     "rs_blend": comp["rs_blend"], "chg": t["price_change_24h"],
                     "z_ath": round(t["price_change_24h"] / 5, 2), "factors": comp})
    rows.sort(key=lambda x: (-x["conviction"], -x["factors"].get("quality", 0)))
    payload = {"as_of": "2026-08-04T19:10:00Z", "benchmark": "SPY",
               "universe": len(rows) + 1, "mcapsum": mcsum, "top10": rows[:10], "all": rows}
    os.makedirs("ledger", exist_ok=True)
    with open("ledger/index.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote ledger/index.json ({len(rows)} names)")
    for r in rows[:10]:
        print(f"  {r['symbol']:7} {r['conviction']:3d} {r['signal']:7} chg={r['chg']:+.1f}% turn={r['turnover']*100:.1f}% rs={r['rs_blend']:+.1f}")


if __name__ == "__main__":
    main()
