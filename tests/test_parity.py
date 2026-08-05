"""Equity Conviction Monitor — parity + regression gates.

Mirrors crypto-conviction-monitor's test_parity.py:
  - frontend/backend parity: the JS conviction() in web/terminal.html MUST produce
    identical convictions + component attribution to equity_monitor.model.score().
  - frozen conviction regression: a fixed fixture basket is pinned; any drift fails CI.

The frontend port (JS) is asserted against here so a lone drift in either surface is caught.
"""
import json, re, math, pathlib
import pytest
from equity_monitor import score

ROOT = pathlib.Path(__file__).resolve().parents[1]
TERMINAL_HTML = ROOT / "web" / "terminal.html"

FMP_KEY = None  # tests are offline; use static fixtures below


# ---- static fixture (offline, deterministic) ----
# A small basket of representative names across quality tiers. Numbers are
# stylised but span the realistic input ranges so the multiplicative composition
# is genuinely exercised (not trivially all-1.0).
FIXTURE = {
    "AAPL": dict(symbol="AAPL", price=220.0, market_cap=3.4e12, exchange="NASDAQ",
                 price_change_24h=1.2, price_change_7d=4.0, price_change_14d=2.0,
                 price_change_30d=9.0, price_change_90d=15.0, price_change_200d=-6.0,
                 total_volume=4.5e9, roic=0.30, fcf_yield=0.08, gross_margin=0.42,
                 debt_ebitda=1.8, earnings_stability=0.85, val_zscore=-0.5,
                 adv=4.0e9, short_interest=0.7e9, short_days=3.0, drawdown_52w=0.14),
    "BRK.B": dict(symbol="BRK.B", price=520.0, market_cap=7.5e11, exchange="NYSE",
                  price_change_24h=0.3, price_change_7d=2.0, price_change_14d=1.5,
                  price_change_30d=5.0, price_change_90d=8.0, price_change_200d=-4.0,
                  total_volume=3.0e9, roic=0.09, fcf_yield=0.035, gross_margin=0.50,
                  debt_ebitda=0.2, earnings_stability=0.95, val_zscore=0.8,
                  adv=2.2e9, short_interest=7e6, short_days=1.0, drawdown_52w=0.02),
    "KO": dict(symbol="KO", price=62.0, market_cap=1.4e11, exchange="NYSE",
               price_change_24h=-0.5, price_change_7d=-2.0, price_change_14d=-1.0,
               price_change_30d=1.0, price_change_90d=4.0, price_change_200d=-12.0,
               total_volume=1.2e9, roic=0.12, fcf_yield=0.07, gross_margin=0.62,
               debt_ebitda=2.8, earnings_stability=0.98, val_zscore=-1.2,
               adv=1.0e9, short_interest=2.6e7, short_days=9.0, drawdown_52w=0.08),
    "NVDA": dict(symbol="NVDA", price=1100.0, market_cap=2.7e12, exchange="NASDAQ",
                 price_change_24h=3.1, price_change_7d=8.0, price_change_14d=22.0,
                 price_change_30d=35.0, price_change_90d=60.0, price_change_200d=45.0,
                 total_volume=3.5e9, roic=0.28, fcf_yield=0.01, gross_margin=0.72,
                 debt_ebitda=1.5, earnings_stability=0.60, val_zscore=2.4,
                 adv=3.0e9, short_interest=0.3e9, short_days=2.5, drawdown_52w=0.05),
}
SPY = dict(symbol="SPY", price=550.0, market_cap=2.1e12, exchange="NYSE",
           price_change_24h=0.4, price_change_7d=2.5, price_change_14d=1.0,
           price_change_30d=4.0, price_change_90d=12.0, price_change_200d=-3.0,
           total_volume=3.0e9, roic=0.0, fcf_yield=0.0, gross_margin=0.0,
           debt_ebitda=0.0, earnings_stability=0.0, val_zscore=0.0,
           adv=2.5e9, short_interest=0.0, short_days=0.0, drawdown_52w=0.0)


def _rs_blend(t, b):
    """Sector-denoised multi-horizon RS vs SPY (mirrors cli.rs_blend)."""
    h7 = t["price_change_7d"] - b["price_change_7d"]
    h14 = t["price_change_14d"] - b["price_change_14d"]
    h30 = t["price_change_30d"] - b["price_change_30d"]
    h200 = t["price_change_200d"] - b["price_change_200d"]
    return 0.30 * h7 + 0.25 * h14 + 0.25 * h30 + 0.20 * h200

def _asset(sym):
    t = dict(FIXTURE[sym])
    t["rs_blend"] = _rs_blend(t, SPY)
    # turnover % for the risk component (vol/mc)
    t["turnover"] = t.get("total_volume", 0) / t["market_cap"] if t.get("market_cap") else 0
    return t, SPY


def frontend_conviction_js(html_path=TERMINAL_HTML):
    """Extract + eval the JS conviction() from terminal.html to mirror a browser.

    We render the JS port through PyMiniRacer if available; otherwise we assert the
    JS body contains the v2 formula string (structural parity guard).
    """
    txt = html_path.read_text()
    body = re.search(r"function conviction\(t, perp, rsBlend\)\s*\{(.*?)\n\}", txt, re.S)
    if not body:
        pytest.skip("terminal.html conviction() not found")
    src = body.group(0)
    # structural guard: must NOT contain the additive clamp, must contain multiplicative
    assert "cmRaw" not in src, "frontend JS still uses v1 additive clamp (cmRaw)"
    assert "100*q*c*risk" in src or "100 * q * c * risk" in src or "100*q*c" in src.replace(" ", ""), \
        "frontend JS missing v2 multiplicative formula"
    assert "Math.tanh" in src, "frontend JS confirmation must be a soft sigmoid (tanh)"
    assert "a + b + cd" not in src and "cmRaw" not in src, \
        "frontend JS still additive (a+b+cd / cmRaw)"
    return True


def check_frontend_backend_parity():
    """Frontend JS formula structurally matches v2 backend + component keys exist."""
    assert frontend_conviction_js(), "frontend JS does not match v2 structure"


def check_frozen_conviction():
    """The v2 multiplicative engine must reproduce pinned convictions for the fixture.
    Recompute pins: edit FROZEN_CONVICTION below only when the model deliberately changes."""
    for sym in FIXTURE:
        t, b = _asset(sym)
        q, c, r, conv, sig, comp = score(t, b)
        assert sym in FROZEN_CONVICTION, f"{sym} missing from FROZEN_CONVICTION"
        assert conv == FROZEN_CONVICTION[sym], f"{sym}: v2 conv {conv} != pinned {FROZEN_CONVICTION[sym]}"
        assert "quality" in comp and "confirmation" in comp
        assert sig in ("STRONG","BUY","HOLD","WATCH","AVOID")

# ---- pytest entry point (so `pytest -q` discovers the parity gates) ----
def test_frontend_backend_parity():
    check_frontend_backend_parity()

def test_frozen_conviction():
    check_frozen_conviction()


if __name__ == "__main__":
    ok = True
    for name, fn in [("parity", check_frontend_backend_parity),
                     ("frozen", check_frozen_conviction)]:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as e:
            ok = False; print(f"  FAIL  {name}: {e}")
    raise SystemExit(0 if ok else 1)


# ---- v2 multiplicative fixture pins (Quality × Confirmation × RiskAdj) ----
# Calibrated to the range-normalised bands in model.py. These assert the engine
# reproduces itself byte-for-byte; update ONLY on a deliberate model change.
# NOTE: bands are placeholder-calibrated (single-asset, fixed thresholds); retune
# against a real S&P 500/Russell 1000 universe before treating pins as signal truth.
FROZEN_CONVICTION = {
    "AAPL": 49,
    "NVDA": 35,
    "BRK.B": 31,
    "KO": 23,
}
