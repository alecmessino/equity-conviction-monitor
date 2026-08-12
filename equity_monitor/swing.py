"""Swing layer: quality-gated mean reversion over a 5-to-25 session horizon.

The conviction score answers "is this a good business I can own". This module answers a
different, shorter question: **this name has fallen, is the bounce worth taking, and what
does it pay against what it risks.** The two are deliberately separate. A name can be a
fine long-term hold and a bad swing, and the reverse.

Three things here are not in the strategy as usually written down, and each exists
because the arithmetic of the usual version does not work:

**Reward-to-risk is the gate, not the drawdown.** A fixed "-15% off the high" entry paired
with a 3xATR stop is a losing structure, and it is losing by construction rather than by
luck. Volatility scales with drawdown, so the stop widens roughly in step with the fall,
while a fixed-fraction retracement target grows faster than linearly. Measured across the
1016-name board, the median name 10-15% off its high offers a 38.2% retracement of +5.3%
against a 3xATR stop of -9.8% — 0.53 to one, needing a 65% strike rate merely to break
even. The same arithmetic at 30-40% off pays 1.35 to one. So the depth threshold is an
*output* of requiring a reward-to-risk floor, not an input to be guessed.

**Drawdown is measured against the benchmark as well as against the high.** A name 25% off
its high while the index is 20% off its own is not a mispriced business; it is beta, and
it will mean-revert with the market or not at all. ``rel_drawdown_52w`` is the drawdown of
the *stock/benchmark ratio*, so it isolates the part of the fall that belongs to this name.
It is a ratio of two observed series and needs no estimated beta — there is no regression
to silently degrade when a name's history is short.

**The retracement leg is anchored to the high that started it.** The swing low is the
lowest low *since* the 52-week high, not the 52-week low, which can predate the high
entirely. Anchoring to ``lo_52w`` produces a leg that never happened, and a target derived
from it is a number about nothing.

One measure deliberately *not* used as a target: the 50-day moving average. It is a
10-week reference being asked to price a 52-week decline. On the current board the median
name 20-25% off its high sits **above** its own 50-day average — the "target" is behind
the price — and across every band the median touch is worth 0.4% to 5.3%. It is reported
because it is a useful state variable, and it is not what the position is sized against.
"""
from __future__ import annotations

import math

# Horizon the whole layer is scoped to. Past this a position is a holding, not a swing,
# and the exit belongs to the conviction model rather than to this one.
MAX_HOLD_DAYS = 25
MIN_HOLD_DAYS = 5

# Retracement fractions of the selloff leg. Target 1 trims, target 2 closes.
FIB_1, FIB_2 = 0.382, 0.5

# Stop distance in ATR(14) multiples. Wilder ATR uses real highs and lows, so this is a
# true range multiple and not a close-to-close approximation.
STOP_ATR_MULT = 3.0

# The gate that replaces a hand-picked drawdown threshold. 1.5 is the point at which a
# coin-flip strike rate is still profitable after costs, with room to be wrong about the
# strike rate — which, on this sample size, one should assume one is.
MIN_REWARD_RISK = 1.5

# Floors that remain, because reward-to-risk alone admits two degenerate cases: a barely
# moved name whose stop happens to be tight, and a name so broken the "leg" is its whole
# history.
MIN_DRAWDOWN = 0.10
MAX_DRAWDOWN = 0.60

# Oversold triggers. Either may fire; requiring both halves the candidate count without
# evidence that the intersection is better.
RSI_ENTRY = 35.0
Z_ENTRY = -1.5

# Quality gate, as a **rank within the universe**, not as a raw q_raw value. q_raw is a
# weighted mean of five percentiles, and a mean of percentiles is not itself uniform — it
# concentrates hard around 0.5. On the current board the 65th percentile of q_raw is
# 0.563, while the literal cut q_raw >= 0.65 keeps 16% of names, not 35%. Writing the gate
# as a percentile of the score distribution is what makes "top third by quality" mean that.
QUALITY_GATE_PCTILE = 0.65


def rsi(closes: list[float], window: int = 14) -> float | None:
    """Wilder's RSI. None when the series is too short to seed the average.

    Wilder smoothing, not a simple mean of gains: the simple version steps discontinuously
    as a large move leaves the window, which puts a name in and out of an oversold trigger
    on a day nothing happened to it.
    """
    if len(closes) < window + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window
    for i in range(window, len(gains)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def stdev(values: list[float], window: int = 50) -> float | None:
    """Sample standard deviation over the trailing window."""
    tail = values[-window:]
    if len(tail) < max(10, window // 2):
        return None
    mean = sum(tail) / len(tail)
    var = sum((v - mean) ** 2 for v in tail) / (len(tail) - 1)
    return math.sqrt(var)


def zscore(price: float, ma: float | None, sd: float | None) -> float | None:
    """How many standard deviations price sits from its moving average."""
    if ma is None or not sd or sd <= 0:
        return None
    return (price - ma) / sd


def selloff_leg(highs: list[float], lows: list[float], window: int = 252) -> dict | None:
    """The decline currently in progress: its high, its low, and when each happened.

    The high is the highest high of the trailing window. The low is the lowest low
    **after** that high — which is the whole point. ``lo_52w`` can sit before the high, in
    which case the implied leg runs backwards through time and every retracement level
    computed from it is fictional.
    """
    if not highs or not lows or len(highs) != len(lows):
        return None
    tail_hi = highs[-window:]
    offset = len(highs) - len(tail_hi)
    peak_local = max(range(len(tail_hi)), key=lambda i: tail_hi[i])
    peak_idx = offset + peak_local
    peak = highs[peak_idx]

    trough_idx = peak_idx
    trough = lows[peak_idx]
    for i in range(peak_idx, len(lows)):
        if lows[i] < trough:
            trough, trough_idx = lows[i], i
    if peak <= 0 or trough <= 0 or peak <= trough:
        return None
    return {
        "leg_high": peak,
        "leg_low": trough,
        "days_since_high": len(highs) - 1 - peak_idx,
        "days_since_low": len(lows) - 1 - trough_idx,
        "leg_depth": (peak - trough) / peak,
    }


def relative_drawdown(closes: list[float], bench_closes: list[float],
                      window: int = 252) -> float | None:
    """Drawdown of the stock/benchmark ratio from its trailing high.

    Isolates the fall that belongs to this name rather than to the market. Beta-free by
    construction: it is a ratio of two observed series, so there is no regression to go
    stale on a short history or to be distorted by the very selloff being measured.

    Positive means the name has underperformed off its own relative peak. A name whose
    absolute drawdown is entirely market-driven scores near zero here.
    """
    n = min(len(closes), len(bench_closes))
    if n < 30:
        return None
    ratio = [closes[-n + i] / bench_closes[-n + i]
             for i in range(n) if bench_closes[-n + i] > 0]
    if len(ratio) < 30:
        return None
    tail = ratio[-window:]
    peak = max(tail)
    if peak <= 0:
        return None
    return (peak - tail[-1]) / peak


def targets(leg: dict, price: float, atr: float | None) -> dict:
    """Retracement targets, stop, and what the trade pays against what it risks.

    Reward-to-risk is computed against **target 1**, the level that trims half the
    position, not against the fuller target 2. Sizing a position on the exit you reach
    least often is how a strategy with an honest edge still loses money.
    """
    lo, hi = leg["leg_low"], leg["leg_high"]
    span = hi - lo
    t1 = lo + FIB_1 * span
    t2 = lo + FIB_2 * span
    out = {
        "target_1": t1,
        "target_2": t2,
        "upside_1": (t1 - price) / price if price > 0 else None,
        "upside_2": (t2 - price) / price if price > 0 else None,
        # Where price sits in the leg: 0 at the low, 1 back at the high.
        "rebound_progress": (price - lo) / span if span > 0 else None,
        "stop": None,
        "stop_pct": None,
        "reward_risk": None,
    }
    if atr and atr > 0 and price > 0:
        stop = price - STOP_ATR_MULT * atr
        risk = price - stop
        out["stop"] = stop
        out["stop_pct"] = risk / price
        if risk > 0 and out["upside_1"] is not None:
            out["reward_risk"] = (t1 - price) / risk
    return out


def evaluate(row: dict, closes: list[float], highs: list[float], lows: list[float],
             bench_closes: list[float] | None = None,
             quality_pctile: float | None = None) -> dict:
    """Every swing field for one name. Pure given its inputs.

    ``row`` supplies the price features the nightly build already derives (price, ma50,
    atr14). ``quality_pctile`` is the name's rank in the universe's q_raw distribution and
    is computed cross-sectionally by the caller, exactly as the model's other percentiles
    are — this module never ranks anything itself.
    """
    price = row.get("price")
    out: dict = {
        "rsi14": None, "sd50": None, "z50": None, "stretch_atr": None,
        "rel_drawdown_52w": None, "leg_high": None, "leg_low": None,
        "leg_depth": None, "days_since_high": None, "days_since_low": None,
        "target_1": None, "target_2": None, "upside_1": None, "upside_2": None,
        "rebound_progress": None, "stop": None, "stop_pct": None, "reward_risk": None,
        "stabilizing": None, "swing_signal": "NONE", "swing_block": None,
    }
    if not price or price <= 0 or not closes:
        out["swing_block"] = "no price"
        return out

    out["rsi14"] = rsi(closes)
    sd = stdev(closes, 50)
    out["sd50"] = sd
    out["z50"] = zscore(price, row.get("ma50"), sd)
    atr = row.get("atr14")
    # ATR-normalised stretch alongside the price-level z. The two disagree for a name
    # whose 50-day window straddles a gap: sd50 of *levels* is inflated by the step and
    # reports the name as un-stretched on the day it fell the most.
    if atr and atr > 0 and row.get("ma50"):
        out["stretch_atr"] = (price - row["ma50"]) / atr

    if bench_closes:
        out["rel_drawdown_52w"] = relative_drawdown(closes, bench_closes)

    leg = selloff_leg(highs, lows)
    if not leg:
        out["swing_block"] = "no leg"
        return out
    out.update({k: leg[k] for k in
                ("leg_high", "leg_low", "leg_depth", "days_since_high", "days_since_low")})
    out.update(targets(leg, price, atr))

    # Stabilisation: has the name stopped making new lows. A stretched name still setting
    # lows daily is the falling-knife case the quality gate is assumed to have removed and
    # demonstrably has not — on the current board, quality is flat across every drawdown
    # band, so it is not selecting for names that have stopped falling.
    out["stabilizing"] = leg["days_since_low"] >= 2

    out["swing_signal"], out["swing_block"] = _classify(row, out, quality_pctile)
    return out


def _classify(row: dict, s: dict, quality_pctile: float | None) -> tuple[str, str | None]:
    """Signal tier, plus the first gate that failed when there is no signal.

    Naming the blocking gate is the difference between a scanner and a black box: "127
    names looked at, 84 failed reward-to-risk" is actionable, "3 candidates" is not.
    """
    dd = row.get("drawdown_52w") or 0.0
    if dd < MIN_DRAWDOWN:
        return "NONE", "drawdown below floor"
    if dd > MAX_DRAWDOWN:
        return "NONE", "drawdown beyond ceiling"
    if quality_pctile is None or quality_pctile < QUALITY_GATE_PCTILE:
        return "NONE", "quality below gate"

    rr = s.get("reward_risk")
    if rr is None:
        return "NONE", "reward:risk unmeasurable"
    if rr < MIN_REWARD_RISK:
        return "NONE", "reward:risk below floor"

    oversold = ((s.get("rsi14") is not None and s["rsi14"] <= RSI_ENTRY)
                or (s.get("z50") is not None and s["z50"] <= Z_ENTRY))
    if not oversold:
        return "WATCH", "not oversold"
    if not s.get("stabilizing"):
        return "WATCH", "still making lows"

    # A name that has fallen on its own terms is the thesis; one that has fallen with the
    # market is a bet on the market, taken through a single stock.
    rel = s.get("rel_drawdown_52w")
    if rel is not None and rel >= 0.10 and rr >= 2.0:
        return "PRIME", None
    return "SETUP", None


# ---------------------------------------------------------------------------
# cross-sectional attachment — Python only, mirrors model.prepare's split
# ---------------------------------------------------------------------------
def attach(rows: list[dict], bars: dict, benchmark: str = "SPY") -> list[dict]:
    """Compute every swing field for a scored universe, in place.

    Ranking lives here rather than in ``evaluate`` for the same reason it lives in
    ``model.prepare`` rather than in ``model.score``: the per-name evaluation stays a pure
    function of its own inputs, and there is exactly one place that knows about the
    cross-section. ETFs carry price features for benchmarking and are never swing-scored.
    """
    scoreable = [r for r in rows if r.get("asset_class") != "ETF"
                 and r.get("q_raw") is not None]
    ranks: dict[str, float] = {}
    if scoreable:
        ordered = sorted(scoreable, key=lambda r: r["q_raw"])
        n = len(ordered)
        for i, r in enumerate(ordered):
            ranks[r["symbol"]] = i / (n - 1) if n > 1 else 0.5

    bench = bars.get(benchmark)
    bench_closes = list(bench.close) if bench else None

    for r in rows:
        if r.get("asset_class") == "ETF":
            continue
        b = bars.get(r.get("symbol"))
        if not b:
            r.update(evaluate({"price": None}, [], [], []))
            continue
        r["quality_pctile"] = ranks.get(r["symbol"])
        r.update(evaluate(r, b.close, b.high, b.low, bench_closes,
                          quality_pctile=ranks.get(r["symbol"])))
    return rows


def summary(rows: list[dict]) -> dict:
    """Counts by signal tier and by blocking gate.

    The block histogram is the part worth publishing. "Eight candidates tonight" invites
    the reader to assume the other thousand were unsuitable businesses; "84 cleared
    quality and depth and failed reward-to-risk" says the threshold is doing the work,
    which is a fact about the market that night rather than about the names.
    """
    tiers: dict[str, int] = {}
    blocks: dict[str, int] = {}
    for r in rows:
        if r.get("asset_class") == "ETF":
            continue
        tiers[r.get("swing_signal") or "NONE"] = tiers.get(r.get("swing_signal") or "NONE", 0) + 1
        b = r.get("swing_block")
        if b:
            blocks[b] = blocks.get(b, 0) + 1
    return {"tiers": dict(sorted(tiers.items())),
            "blocks": dict(sorted(blocks.items(), key=lambda kv: -kv[1])),
            "candidates": sorted(
                ({"symbol": r["symbol"], "name": r.get("name"), "sector": r.get("sector"),
                  "conviction": r.get("conviction"), "quality_pctile": r.get("quality_pctile"),
                  "drawdown_52w": r.get("drawdown_52w"),
                  "rel_drawdown_52w": r.get("rel_drawdown_52w"),
                  "rsi14": r.get("rsi14"), "z50": r.get("z50"),
                  "reward_risk": r.get("reward_risk"), "upside_1": r.get("upside_1"),
                  "target_1": r.get("target_1"), "stop": r.get("stop"),
                  "rebound_progress": r.get("rebound_progress"),
                  "signal": r.get("swing_signal")}
                 for r in rows
                 if r.get("swing_signal") in {"PRIME", "SETUP"}),
                key=lambda c: -(c["reward_risk"] or 0))}
