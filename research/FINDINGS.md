# Structural yield harvesting — feasibility measurement

A spike against the four-pillar thesis: cointegration, volatility risk premium,
volatility harvesting, microstructure overreaction. Everything below is measured, on
keyless data, on 2026-08-10. Nothing here is wired into the conviction score.

The sibling crypto spike is at `crypto-conviction-monitor/research/FINDINGS.md`. The
two disagree on pillars 1 and 2, and the disagreements are the informative part: pairs
trading works in crypto and not here, while VRP is cleanly observable in crypto (Deribit
publishes an implied-vol index) and has to be reconstructed from option chains here.

**Headline: two pillars survive measurement, one is unobtainable as specified, and one
is real but does not do what the thesis claims it does.**

| Pillar | Verdict | The number that decided it |
|---|---|---|
| 1. Cointegration | **Fails on equities, marginal on crypto** | 136 large-cap pairs tested, 8 passed at 5%, 6.8 expected by chance. After FDR correction: **0**. Crypto: 45 tested, 4 passed, 2 survive correction. |
| 2. VRP | **Real, and the strongest of the four** | Index VRP positive on **86.3%** of 439 overlapping windows, mean +4.0 vol points. Crypto DVOL−RV: BTC +8.1, ETH +10.1 points today. |
| 3. Vol harvesting | **Theorem holds exactly; the claim built on it does not** | Rebalanced/geometric = 1.0573 vs predicted 1.0571. Rebalanced/buy-and-hold = **0.9916**. |
| 4. Microstructure | **Not measurable at this sample size** | 89 three-sigma events across 17 names in 2y ≈ 2.6 per name per year. Pooled forward 5-day return **−0.15%, win rate 52%** — a coin flip. |

---

## Pillar 1 — Cointegration does not survive multiple-testing correction

The thesis names Visa/Mastercard as the canonical equity pair. It is not cointegrated.
Neither is any other economically motivated pair on a 504-day window:

| Pair | ADF | Half-life | Verdict |
|---|---|---|---|
| V / MA | −1.90 | 50 d | fail |
| KO / PEP | −0.76 | 119 d | fail |
| HD / LOW | −1.54 | 73 d | fail |
| XOM / CVX | −2.46 | 30 d | fail |
| GS / MS | −2.94 | 20 d | fail (closest) |
| UNP / CSX | −2.61 | 24 d | fail |
| UPS / FDX | −1.60 | 73 d | fail |
| AAPL / MSFT | −1.31 | 88 d | fail |

Searching all 136 pairs instead returns 8 "passers" — against 6.8 expected from noise
alone at the 5% level. They are indistinguishable from chance, and they look it: the
best is **PEP/V** (ADF −4.63), a pairing with no shared economics, and **6 of the 8 have
negative hedge ratios**, meaning they are not spreads at all but long-both or short-both
directional bets wearing a market-neutral label.

Applying Benjamini-Hochberg across the search, the cutoff tightens from −3.34 to −4.70
and **zero equity pairs survive**.

Crypto is genuinely better. 45 pairs, cutoff tightens to −4.27, and two survive with
positive hedge ratios and actionable half-lives:

```
DOT/XRP   adf −4.46   beta 1.51   half-life  6 d   z +0.13
LTC/XRP   adf −4.39   beta 0.97   half-life  6 d   z +2.19   <- at entry threshold
```

Note that **BTC/ETH, the thesis's crypto example, fails** (ADF −2.80). Note also that
both survivors contain XRP, so the two tests are not independent and BH is optimistic
here; a common-factor check belongs in front of any capital.

**Design consequence.** A pairs pillar must apply an FDR correction sized to the search,
reject negative hedge ratios outright, and bound half-life. Without those three gates it
is a data-mining machine that reports noise as signal. All three are implemented in
`structural_yield.py` and pinned by `test_trap_*` tests.

## Pillar 2 — VRP is real, is an index phenomenon, and is currently thin

Measured correctly — implied vol on day *t* against volatility realised over the **next**
21 sessions, not the previous 21:

```
VIX vs forward 21d SPY RV, 439 overlapping windows
  mean spread     +4.0 vol points
  median          +5.5
  positive        86.3% of windows
  worst          −35.3 vol points
```

That is a structural premium with a genuine tail — exactly the shape of an insurance
book. Right now it is compressed: VIX 14.9 against 21-day realised 14.0, a spread of
**+0.9 points** against a +4.0 mean. The premium is near the bottom of its range.

**The per-name result is a trap worth documenting.** Naively comparing 25-day ATM IV
against *trailing* 21-day realised vol produced apparently massive negative premiums:

| | ATM IV (25d) | trailing RV21 | naive "VRP" | RV ex-largest-day |
|---|---|---|---|---|
| MSFT | 28.1% | 56.5% | −28.4 pts | **30.7%** |
| AAPL | 24.1% | 37.0% | −13.0 pts | **25.3%** |
| KO | 18.9% | 28.4% | −9.4 pts | **23.4%** |
| SPY | 12.6% | 14.0% | −1.4 pts | 13.0% |

MSFT's trailing window contained a single +14.4% earnings day (2026-07-30) worth 26 vol
points on its own. A 25-day option cannot contain that jump — the next report is a
quarter away. SPY, having no single-name earnings, barely moves under the adjustment.

Corrected, single-name VRP is roughly **zero to slightly negative** for mega-caps while
index VRP is structurally positive. That asymmetry is not a defect in the measurement —
it *is* the documented correlation risk premium, and it is the reason dispersion trading
exists. The thesis files dispersion under pillar 3; the data says it belongs here, as
the equity expression of pillar 2: sell index vol, buy single-name vol.

**Obtainability, keyless:** CBOE's delayed-quote endpoint returns full per-name chains
with IV and greeks (`cdn.cboe.com/api/global/delayed_quotes/options/{SYM}.json`, 0.5–1.6
MB, 1.2k–3.5k contracts, ~16–24 expiries). Deribit's DVOL index gives BTC/ETH implied
vol with two years of history. Both are unauthenticated. Per-name chains do not scale to
Russell 1000 (~1 GB/night) and would need a curated subset.

## Pillar 3 — The theorem is exact; the conclusion drawn from it is wrong

Fernholz's excess growth rate on an equal-weight top-10 crypto basket, 365 days:

```
gamma* (excess growth)              0.0555 / yr
rebalanced / weighted-avg-log-growth  1.0573   (theorem predicts 1.0571)
rebalanced / buy-and-hold             0.9916
```

The identity holds to four decimal places — the implementation is correct and the 5.55%
harvest is genuinely captured. **And rebalancing still lost 0.84% to doing nothing.**

Both are true because the theorem's benchmark is the *weighted average of the
constituents' log growth rates*, not buy-and-hold. Buy-and-hold lets winners compound
their weight; constant-weight rebalancing systematically sells them. Over this window
BNB returned 0.745× while the basket median returned ~0.39×, and trimming it cost more
than the harvest paid. The equity basket, being far less dispersed, did show rebalancing
ahead (+3.25% daily, +1.96% quarterly, on 2.24× annual turnover).

So the thesis's claim — "compounding geometric returns independently of macro
direction" — is not supported. The harvest is real and always non-negative *against the
right benchmark*, but whether it beats the obvious alternative depends entirely on cross-
sectional dispersion and trend. It must be reported against both benchmarks, never one.

Turnover matters too: 2.16× notional per year at daily frequency to harvest 5.6% gross.
At 10 bps round-trip that is 22 bps of drag; at 50 bps the harvest is largely gone.

## Pillar 4 — Real, thin, and not uniform

Three-sigma down-days against a trailing 60-day window: **89 events across 17 names over
two years**, ≈2.6 per name per year.

Pooled across every event, the forward 5-day return is **−0.15% with a 52% win rate**.
That is a coin flip. The per-name numbers look far more exciting — HD +3.1% (on 4
events), FDX +3.2% (6), against CVX −2.7% (7) and CSX −2.7% (5) — and every one of them
is noise: 2–8 observations cannot distinguish a 3% edge from zero.

The honest reading is that this sample cannot size the pillar, not that the pillar is
absent. The published research on it conditions on far more than a sigma threshold —
liquidity, the absence of an information event, sector-relative moves. Testing it
properly needs a pooled, universe-wide event study over years, which is precisely the
dataset `ledger/snapshots/` is accumulating and cannot be reconstructed after the fact.
Until then this pillar should report events, not signals.

---

## The structural point: these are not four independent edges

Pairs trading loses when a spread trends. Put-writing loses when the market gaps.
Constant-weight rebalancing loses when a constituent trends to zero. Dip-buying loses
when the dip was information. **All four are short convexity — they are four expressions
of one trade: sell insurance, collect premium, lose in the tail.**

Sizing them as though they diversify each other is the mechanism by which this strategy
class fails. The measured −35.3 vol-point worst case on index VRP is the same event that
would gap a pairs spread, force a rebalance into a falling knife, and turn a three-sigma
dip into a five-sigma one.

`structural_yield.regime_ok()` therefore gates all four on a shared pair of conditions:
premium above a floor, and trend strength below a ceiling. It does not try to remove the
tail risk — that risk *is* the premium — it refuses to add exposure when the environment
already says the tail is opening.

## Why this is not wired into the conviction score

Both repos segment their history by a specification hash so that an Information
Coefficient computed across recorded nights means something. These pillars are
orthogonal to the conviction score — market-neutral yield extraction rather than
cross-sectional asset ranking — and folding them in would silently re-base every night
already recorded, destroying the only asset either repo has been accumulating.

The recommendation is a **parallel engine**: its own ledger, its own spec hash, its own
track record, sharing the price fetch and the regime gate. On the equity side this costs
no new data at all — `prices.load()` already returns 521 daily bars per name inside the
nightly run, which is everything pillars 1, 3 and 4 need.

## Reproducing

```bash
python3 -m pytest tests/test_structural_yield.py -q     # 18 property + trap tests
python3 research/probe.py                               # re-measures everything above
```
