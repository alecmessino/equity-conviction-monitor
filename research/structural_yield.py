"""Structural yield harvesting — the four pillars, as measurable quantities.

This is a **research spike, deliberately not wired into the conviction score.** The
board's history is segmented by ``model.spec_hash()`` so that an Information
Coefficient computed across it means something; folding a second, orthogonal engine
into that score would silently re-base every night already recorded. These pillars
therefore compute alongside the board and would write their own ledger under their
own hash. See research/FINDINGS.md for the measurements that shaped every choice here.

The four pillars, and what each actually is:

1. **Cointegration.** Two assets share a stochastic trend, so a linear combination of
   their log prices is stationary and its deviations are tradeable. Tested by
   Engle-Granger: regress one log price on the other, then test the residual for a
   unit root.

2. **Volatility risk premium.** Implied volatility exceeds subsequently-realised
   volatility because someone is paying to be insured. Measured as IV minus *forward*
   RV, never IV minus trailing RV -- see ``realized_vol`` for why that distinction is
   the whole ballgame.

3. **Volatility harvesting.** Fernholz's excess growth rate. A theorem, not a
   backtest: strictly positive for any diversified long-only weighting. What it is
   positive *relative to* is the subtle part, and is the single most misunderstood
   thing in this file -- see ``excess_growth``.

4. **Microstructure overreaction.** A multi-sigma move without a fundamental catalyst
   partially reverts, because the move was liquidity demand rather than information.

Pure stdlib. The nightly job runs on keyless public data with no scientific stack, and
a pillar that needs numpy is a pillar that does not run.
"""
from __future__ import annotations

import math

# Engle-Granger critical values for the residual of a two-variable cointegrating
# regression with a constant. These are NOT the standard Dickey-Fuller values: the
# residual comes from an estimated relationship, so the test statistic is biased
# toward rejection and the critical values are correspondingly further left. Using
# the textbook -2.86 here would roughly triple the false-positive rate.
EG_CRITICAL = {0.10: -3.04, 0.05: -3.34, 0.01: -3.90}

# A spread is only market-neutral if the hedge ratio is positive. A negative beta
# means the "spread" is long both legs or short both legs -- a levered directional
# bet wearing a pairs trade's clothing. Measured: 6 of the 8 pairs that passed the
# 5% cointegration test on the large-cap sample had negative betas.
MIN_HEDGE_RATIO = 0.0

# Half-life bounds, in trading days. Below the floor the reversion is faster than a
# nightly job can act on and is mostly bid-ask bounce; above the ceiling the capital
# is committed for a quarter to earn one spread crossing, and the cointegrating
# relationship is unlikely to survive that long anyway.
MIN_HALF_LIFE, MAX_HALF_LIFE = 2.0, 45.0


# ---------------------------------------------------------------------------
# linear algebra (small, dense, dependency-free)
# ---------------------------------------------------------------------------
def ols(y: list[float], x: list[float]) -> tuple[float, float]:
    """(alpha, beta) for y = alpha + beta*x. Beta is the hedge ratio."""
    n = len(y)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    if not sxx:
        return my, 0.0
    beta = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
    return my - beta * mx, beta


def _inv(m: list[list[float]]) -> list[list[float]] | None:
    """Gauss-Jordan inverse with partial pivoting. Small k only."""
    k = len(m)
    a = [list(row) + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(m)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        pv = a[col][col]
        a[col] = [v / pv for v in a[col]]
        for r in range(k):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [v - f * w for v, w in zip(a[r], a[col])]
    return [row[k:] for row in a]


def _lstsq(rows: list[list[float]], ys: list[float]):
    k = len(rows[0])
    xtx = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    inv = _inv(xtx)
    if inv is None:
        return None, None, None
    beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]
    resid = [y - sum(b * v for b, v in zip(beta, r)) for r, y in zip(rows, ys)]
    return beta, resid, inv


# ---------------------------------------------------------------------------
# pillar 1: cointegration
# ---------------------------------------------------------------------------
def adf_tstat(series: list[float], lags: int = 1) -> float | None:
    """Augmented Dickey-Fuller t-statistic on gamma in

        d_t = c + gamma * s_{t-1} + sum_i phi_i * d_{t-i}

    More negative means more strongly mean-reverting. Compare against
    ``EG_CRITICAL``, not the textbook DF values.
    """
    d = [series[i] - series[i - 1] for i in range(1, len(series))]
    rows, ys = [], []
    for t in range(lags, len(d)):
        row = [1.0, series[t]]
        row.extend(d[t - i - 1] for i in range(lags))
        rows.append(row)
        ys.append(d[t])
    if not rows or len(rows) < len(rows[0]) + 5:
        return None
    beta, resid, inv = _lstsq(rows, ys)
    if beta is None:
        return None
    dof = len(rows) - len(beta)
    if dof <= 0:
        return None
    var = (sum(r * r for r in resid) / dof) * inv[1][1]
    return beta[1] / math.sqrt(var) if var > 0 else None


def half_life(spread: list[float]) -> float | None:
    """Ornstein-Uhlenbeck half-life in periods. None if the spread is divergent."""
    lag = spread[:-1]
    d = [spread[i] - spread[i - 1] for i in range(1, len(spread))]
    _, beta = ols(d, lag)
    if beta >= 0 or beta <= -1:
        return None
    return math.log(2) / -math.log(1 + beta)


def engle_granger(a: list[float], b: list[float], min_obs: int = 250) -> dict | None:
    """Cointegration test on two aligned price series. ``a`` is the dependent leg.

    Callers must align on trading dates first: two series of the same *length* but
    different *dates* produce a spread that is an artifact of the misalignment.
    """
    n = min(len(a), len(b))
    if n < min_obs:
        return None
    la = [math.log(v) for v in a[-n:] if v > 0]
    lb = [math.log(v) for v in b[-n:] if v > 0]
    if len(la) != n or len(lb) != n:
        return None
    alpha, beta = ols(la, lb)
    spread = [x - (alpha + beta * y) for x, y in zip(la, lb)]
    mu = sum(spread) / n
    sd = math.sqrt(sum((s - mu) ** 2 for s in spread) / (n - 1))
    return {
        "n": n, "intercept": alpha, "hedge_ratio": beta,
        "adf": adf_tstat(spread), "half_life": half_life(spread),
        "z": (spread[-1] - mu) / sd if sd else 0.0, "spread_sd": sd,
    }


def benjamini_hochberg(stats: list[float], alpha: float = 0.05) -> float:
    """The ADF cutoff that holds the false-discovery rate at ``alpha``.

    Searching k pairs for cointegration runs k simultaneous tests, so at the nominal
    5% level k/20 of them pass on noise alone. Measured on the large-cap sample: 136
    pairs tested, 8 passed, 6.8 expected by chance -- the survivors were entirely
    consistent with noise. Without this correction the pillar reports data-mining
    artifacts as signals, which is exactly the failure mode this codebase exists to
    avoid.

    Returns the (more negative) threshold a statistic must beat. Approximated by
    interpolating the Engle-Granger critical values on a log-p scale, which is close
    enough for a gate whose purpose is to be conservative.
    """
    k = len(stats)
    if k <= 1:
        return EG_CRITICAL[alpha]
    # With no discoveries the operative cutoff is the strictest rung, alpha/k: that is
    # what the single best statistic in the search would have to beat. Defaulting to
    # the uncorrected value instead would hand back a *looser* threshold the wider the
    # search got, which is precisely backwards.
    thresh = _eg_critical_at(alpha / k)
    for i, s in enumerate(sorted(stats), start=1):
        # BH: the i-th smallest statistic is compared against level alpha*i/k. Each
        # additional discovery relaxes the cutoff for the ones behind it.
        cut = _eg_critical_at(alpha * i / k)
        if s <= cut:
            thresh = cut
    return thresh


def _eg_critical_at(p: float) -> float:
    """Interpolate an Engle-Granger critical value at an arbitrary level."""
    pts = sorted(EG_CRITICAL.items())          # [(0.01,-3.90),(0.05,-3.34),(0.10,-3.04)]
    if p <= pts[0][0]:
        # Extrapolate left: each decade tighter costs roughly 0.56 of a t-unit.
        return pts[0][1] - 0.56 * math.log10(pts[0][0] / max(p, 1e-6))
    if p >= pts[-1][0]:
        return pts[-1][1]
    for (p0, c0), (p1, c1) in zip(pts, pts[1:]):
        if p0 <= p <= p1:
            f = (math.log(p) - math.log(p0)) / (math.log(p1) - math.log(p0))
            return c0 + f * (c1 - c0)
    return pts[-1][1]


def tradeable(result: dict, adf_cutoff: float) -> bool:
    """Whether a cointegration result describes a spread worth capital."""
    if not result or result.get("adf") is None:
        return False
    hl = result.get("half_life")
    return (result["adf"] <= adf_cutoff
            and result["hedge_ratio"] > MIN_HEDGE_RATIO
            and hl is not None
            and MIN_HALF_LIFE <= hl <= MAX_HALF_LIFE)


# ---------------------------------------------------------------------------
# pillar 2: volatility risk premium
# ---------------------------------------------------------------------------
def realized_vol(closes: list[float], window: int = 21, periods: int = 252,
                 drop_extremes: int = 0) -> float | None:
    """Annualised close-to-close realised volatility.

    ``drop_extremes`` removes the n largest absolute returns before computing. This is
    not cosmetic smoothing: a single-name trailing window that contains an earnings gap
    measures an event the *next* window cannot contain, because the next report is a
    quarter away. Measured on 2026-08-10: MSFT's 21-day RV was 56.5%, of which one day
    (+14.4% on 2026-07-30, earnings) accounted for 26 points -- ex that day, 30.7%.
    Comparing a 25-day implied vol against the contaminated 56.5% reports a VRP of
    -28 points where the tradeable number is roughly -3. SPY, which has no single-name
    earnings, moved only 14.0% -> 13.0% under the same adjustment.
    """
    tail = closes[-(window + 1):]
    if len(tail) < 10:
        return None
    rets = [math.log(tail[i] / tail[i - 1]) for i in range(1, len(tail)) if tail[i - 1] > 0]
    if drop_extremes:
        rets = sorted(rets, key=abs)[:-drop_extremes] or rets
    if len(rets) < 5:
        return None
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var * periods)


def forward_vrp(implied: list[tuple[str, float]], closes: list[float],
                dates: list[str], window: int = 21,
                periods: int = 252) -> dict | None:
    """The premium measured the only way that is not circular: implied vol on day t
    against the volatility actually realised over the ``window`` days *after* t.

    Trailing RV answers "was the option expensive relative to what just happened",
    which is not the question an insurance seller is paid to answer.
    """
    idx = {d: i for i, d in enumerate(dates)}
    spreads = []
    for day, iv in implied:
        i = idx.get(day)
        if i is None or i + window + 1 > len(closes):
            continue
        fwd = realized_vol(closes[i:i + window + 1], window, periods)
        if fwd is not None:
            spreads.append(iv - fwd)
    if not spreads:
        return None
    spreads.sort()
    n = len(spreads)
    return {
        "n": n,
        "mean": sum(spreads) / n,
        "median": spreads[n // 2],
        "pct_positive": sum(1 for s in spreads if s > 0) / n,
        "worst": spreads[0],
        "p05": spreads[max(0, int(0.05 * n) - 1)],
    }


# ---------------------------------------------------------------------------
# pillar 3: volatility harvesting
# ---------------------------------------------------------------------------
def cov_matrix(series: list[list[float]], periods: int = 252) -> list[list[float]]:
    """Annualised covariance of log returns for date-aligned close series."""
    rets = [[math.log(s[i] / s[i - 1]) for i in range(1, len(s)) if s[i - 1] > 0]
            for s in series]
    m = min(len(r) for r in rets)
    rets = [r[-m:] for r in rets]
    means = [sum(r) / m for r in rets]
    n = len(rets)
    return [[sum((rets[i][t] - means[i]) * (rets[j][t] - means[j])
                 for t in range(m)) / (m - 1) * periods
             for j in range(n)] for i in range(n)]


def excess_growth(cov: list[list[float]], w: list[float]) -> float:
    """Fernholz excess growth rate: 0.5 * (sum_i w_i*sigma_ii - w'*Cov*w).

    **Read this before using the number.** The theorem says the continuously
    rebalanced portfolio's log growth exceeds *the weighted average of the
    constituents' log growth rates* by exactly this amount. It does NOT say the
    rebalanced portfolio beats buy-and-hold, and the difference is not academic:
    buy-and-hold lets winners compound their weight, which is a convexity the
    constant-weight portfolio gives up.

    Measured on an equal-weight top-10 crypto basket over 365 days:

        rebalanced / weighted-avg-log-growth = 1.0573   (theorem predicts 1.0571)
        rebalanced / buy-and-hold            = 0.9916

    The theorem held to four decimals *and* rebalancing still lost 0.84% to doing
    nothing. Both statements are true simultaneously. Treat this as a measurement of
    the harvest available, never as a forecast of outperformance.
    """
    n = len(w)
    wcw = sum(w[i] * cov[i][j] * w[j] for i in range(n) for j in range(n))
    return 0.5 * (sum(w[i] * cov[i][i] for i in range(n)) - wcw)


def rebalance_backtest(series: list[list[float]], w: list[float],
                       every: int = 1, cost_bps: float = 0.0) -> dict:
    """Constant-weight rebalancing vs buy-and-hold, with turnover and costs.

    Turnover is reported because the harvest is a gross number and the rebalancing
    that produces it is not free. At daily frequency the crypto basket turned over
    2.16x notional per year to harvest 5.6% -- at 10bps round-trip that is 22bps of
    cost against it, which is survivable; at 50bps it is not.
    """
    m = min(len(s) for s in series)
    series = [s[-m:] for s in series]
    n = len(series)
    units_r = [w[i] / series[i][0] for i in range(n)]
    units_h = list(units_r)
    turnover = 0.0
    cost = cost_bps / 10000.0
    drag = 1.0
    for t in range(1, m):
        if t % every:
            continue
        vals = [units_r[i] * series[i][t] for i in range(n)]
        tot = sum(vals)
        if tot <= 0:
            break
        target = [w[i] * tot for i in range(n)]
        tno = sum(abs(target[i] - vals[i]) for i in range(n)) / tot / 2
        turnover += tno
        drag *= (1 - tno * cost)
        units_r = [target[i] / series[i][t] for i in range(n)]
    reb = sum(units_r[i] * series[i][-1] for i in range(n)) * drag
    hold = sum(units_h[i] * series[i][-1] for i in range(n))
    geo = math.exp(sum(w[i] * math.log(series[i][-1] / series[i][0]) for i in range(n)))
    return {
        "rebalanced": reb, "buy_and_hold": hold, "geometric": geo,
        "vs_hold": reb - hold, "vs_geometric": reb - geo,
        "turnover": turnover, "periods": m,
    }


# ---------------------------------------------------------------------------
# pillar 4: microstructure overreaction
# ---------------------------------------------------------------------------
def sigma_event(closes: list[float], window: int = 60) -> dict | None:
    """How many trailing standard deviations the most recent move was."""
    if len(closes) < window + 2:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            if closes[i - 1] > 0]
    hist = rets[-(window + 1):-1]
    if len(hist) < 10:
        return None
    mu = sum(hist) / len(hist)
    sd = math.sqrt(sum((r - mu) ** 2 for r in hist) / (len(hist) - 1))
    if not sd:
        return None
    return {"ret": rets[-1], "sigma": (rets[-1] - mu) / sd, "daily_vol": sd}


# ---------------------------------------------------------------------------
# the shared regime gate
# ---------------------------------------------------------------------------
# Every pillar above is short convexity. Pairs trading loses when a spread trends,
# put-writing loses when the market gaps, constant-weight rebalancing loses when a
# constituent trends to zero, and dip-buying loses when the dip was information. They
# are not four independent edges -- they are four expressions of one trade: sell
# insurance, collect premium, lose in the tail. Sizing them as if they diversify each
# other is the mechanism by which this class of strategy blows up.
#
# The premium is the compensation for that risk, so the gate does not try to remove it.
# It refuses to *add* exposure when the environment already says the tail is opening.
VRP_FLOOR = 0.02          # IV must exceed forward-RV expectation by 2 vol points
TREND_CEILING = 0.35      # |trend strength| above this is a directional regime


def regime_ok(vrp_spread: float | None, trend_strength: float | None) -> dict:
    """Whether the environment is one these pillars are paid in.

    ``trend_strength`` is a signed, normalised measure of directional persistence
    (e.g. the ratio of net drift to summed absolute daily moves over the window).
    Near zero is chop, which is what all four pillars want.
    """
    reasons = []
    if vrp_spread is None:
        reasons.append("vrp_unobservable")
    elif vrp_spread < VRP_FLOOR:
        reasons.append(f"vrp_thin({vrp_spread:.3f})")
    if trend_strength is None:
        reasons.append("trend_unobservable")
    elif abs(trend_strength) > TREND_CEILING:
        reasons.append(f"trending({trend_strength:+.2f})")
    return {"ok": not reasons, "reasons": reasons}


def trend_strength(closes: list[float], window: int = 60) -> float | None:
    """Signed efficiency ratio in [-1, 1]: net move over summed absolute moves.

    1.0 is a straight line up, 0.0 is pure chop. This is the regime variable the whole
    thesis is conditioned on, so it is measured rather than assumed.
    """
    tail = closes[-(window + 1):]
    if len(tail) < 10:
        return None
    net = tail[-1] - tail[0]
    path = sum(abs(tail[i] - tail[i - 1]) for i in range(1, len(tail)))
    return (net / path) if path else None
