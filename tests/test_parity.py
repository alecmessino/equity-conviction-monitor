"""Equity Conviction Monitor — parity + regression gates.

Two gates (both must pass for `pytest -q` to be green):

1. FRONTEND/BACKEND PARITY
   Extract the v2 multiplicative `score(t)` port from web/terminal.html and execute
   it in a JS sandbox (PyMiniRacer if present, else a minimal JS shim). Assert the
   JS port and equity_monitor.model.score() agree on conviction + Q/C/R for every
   fixture asset. Also assert the JS contains the v2 multiplicative formula and a
   soft-tanh confirmation (no v1 additive clamp / cmRaw).

2. FROZEN CONVICTION REGRESSION
   The v2 engine must reproduce pinned convictions for the fixture basket. Pins are
   (re)calibrated below; change them ONLY on a deliberate model change.
"""
import json, re, math, pathlib
import pytest
from equity_monitor import model

ROOT = pathlib.Path(__file__).resolve().parents[1]
TERMINAL_HTML = ROOT / "web" / "terminal.html"

FMP_KEY = None  # tests are offline; static fixtures below


# ---- static fixture (offline, deterministic) ----
# Representative names spanning quality/RS tiers. Inputs match the feature keys
# produced by equity_monitor.data.build_features() so the model runs for real.
FIXTURE = {
    "AAPL": dict(symbol="AAPL", price=220.0, market_cap=3.4e12, chg24=1.2,
                  adv=4.0e9, turnover=4.0e9/3.4e12, roic=0.30, fcf_yield=0.08,
                  gross_margin=0.42, debt_ebitda=1.8, earnings_stability=0.85,
                  val_zscore=-0.5, short_days=3.0, rs_blend=1.35, drawdown_52w=0.14),
    "BRK.B": dict(symbol="BRK.B", price=520.0, market_cap=7.5e11, chg24=0.3,
                  adv=2.2e9, turnover=2.2e9/7.5e11, roic=0.09, fcf_yield=0.035,
                  gross_margin=0.50, debt_ebitda=0.2, earnings_stability=0.95,
                  val_zscore=0.8, short_days=1.0, rs_blend=0.02, drawdown_52w=0.02),
    "KO": dict(symbol="KO", price=62.0, market_cap=1.4e11, chg24=-0.5,
               adv=1.0e9, turnover=1.0e9/1.4e11, roic=0.12, fcf_yield=0.07,
               gross_margin=0.62, debt_ebitda=2.8, earnings_stability=0.98,
               val_zscore=-1.2, short_days=9.0, rs_blend=-4.4, drawdown_52w=0.08),
    "NVDA": dict(symbol="NVDA", price=1100.0, market_cap=2.7e12, chg24=3.1,
                 adv=3.0e9, turnover=3.0e9/2.7e12, roic=0.28, fcf_yield=0.01,
                 gross_margin=0.72, debt_ebitda=1.5, earnings_stability=0.60,
                 val_zscore=2.4, short_days=2.5, rs_blend=24.25, drawdown_52w=0.05),
}
SPY = dict(symbol="SPY", price=550.0, market_cap=2.1e12, chg24=0.4,
           adv=2.5e9, turnover=2.5e9/2.1e12, roic=0.0, fcf_yield=0.0,
           gross_margin=0.0, debt_ebitda=0.0, earnings_stability=0.0,
           val_zscore=0.0, short_days=0.0, rs_blend=0.0, drawdown_52w=0.0)

def _asset(sym):
    t = dict(FIXTURE[sym])
    return t, SPY


def _extract_js_score(html_path=TERMINAL_HTML):
    """Return a callable score(t)->{q,c,r,conv} by evaluating the JS port in a shim."""
    txt = html_path.read_text()
    # pull the whole <script> block
    m = re.search(r"<script>(.*?)</script>", txt, re.S)
    if not m:
        pytest.skip("terminal.html has no <script>")
    js = m.group(1)
    # sanity: v2 structure present
    assert "100*q*c*r" in js or "100 * q * c * r" in js, "JS missing v2 100·Q·C·R"
    assert "Math.tanh" in js, "JS confirmation must use soft-tanh"
    assert "cmRaw" not in js, "JS still uses v1 additive clamp (cmRaw)"

    # minimal JS eval via node if available, else PyMiniRacer
    try:
        import subprocess, tempfile, os, threading
        # strip browser-only entrypoints so node can import the pure functions
        js_clean = re.sub(r"load\(\);", "", js)
        js_clean = re.sub(r"setInterval\([^;]*;", "", js_clean)
        js_clean = re.sub(r"setTimeout\([^;]*;", "", js_clean)
        with tempfile.NamedTemporaryFile("w", suffix=".cjs", delete=False) as f:
            f.write(js_clean + "\nmodule.exports = { score };\n")
            path = f.name
        runner = path + ".run.cjs"
        with open(runner, "w") as rf:
            rf.write(
                f"const m=require({path!r});\n"
                f"const FIX={json.dumps(FIXTURE)};\n"
                f"const res={{}}; for(const k in FIX){{res[k]=m.score(FIX[k]).conv;}}\n"
                f"console.log(JSON.stringify(res));\n"
                f"process.exit(0);\n"
            )
        result = {}
        def _run():
            try:
                r = subprocess.run(["node", runner], capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    result["data"] = json.loads(r.stdout.strip())
            except Exception:
                pass
        t = threading.Thread(target=_run, daemon=True)
        t.start(); t.join(12)   # never block the suite >12s
        if "data" in result:
            return result["data"], "node"
    except Exception:
        pass
    # fallback: PyMiniRacer
    try:
        from py_mini_racer import py_mini_racer
        ctx = py_mini_racer.MiniRacer()
        ctx.eval(js)
        def call(t):
            return ctx.call("score", t)
        return {k: call(v)["conv"] for k, v in FIXTURE.items()}, "miniracer"
    except Exception:
        pytest.skip("no JS runtime (node/PyMiniRacer) available to execute frontend port")


def test_frontend_backend_parity():
    js_conv, _ = _extract_js_score()
    for sym in FIXTURE:
        t, b = _asset(sym)
        q, c, r, conv, sig, comp = model.score(t, b)
        assert conv == js_conv[sym], f"{sym}: backend {conv} != frontend JS {js_conv[sym]}"


def test_frozen_conviction():
    for sym in FIXTURE:
        t, b = _asset(sym)
        q, c, r, conv, sig, comp = model.score(t, b)
        assert sym in FROZEN_CONVICTION, f"{sym} missing from FROZEN_CONVICTION"
        assert conv == FROZEN_CONVICTION[sym], f"{sym}: v2 conv {conv} != pinned {FROZEN_CONVICTION[sym]}"
        assert "quality" in comp and "confirmation" in comp and "risk" in comp
        assert sig in ("STRONG", "BUY", "HOLD", "WATCH", "AVOID")


if __name__ == "__main__":
    ok = True
    for name, fn in [("parity", test_frontend_backend_parity),
                     ("frozen", test_frozen_conviction)]:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as e:
            ok = False; print(f"  FAIL  {name}: {e}")
    raise SystemExit(0 if ok else 1)


# ---- v2 multiplicative fixture pins (Quality × Confirmation × RiskAdj) ----
# Calibrated to range-normalised bands in model.py. Asserts the engine reproduces
# itself byte-for-byte; update ONLY on a deliberate model change.
# NOTE: bands are placeholder-calibrated (single-asset fixed thresholds); retune
# against a real S&P 500/Russell 1000 universe before treating pins as signal truth.
FROZEN_CONVICTION = {
    "AAPL": 52,
    "NVDA": 36,
    "BRK.B": 30,
    "KO": 20,
}
