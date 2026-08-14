# Equity Conviction Monitor

A quality mean-reversion terminal for US large caps, built entirely on **keyless public
data**: fundamentals from SEC EDGAR XBRL filings, prices from Nasdaq, macro from FRED.

**Live:** https://alecmessino.github.io/equity-conviction-monitor/web/terminal.html

```
Conviction = 100 × (Q · C · R)^(1/3)
```

| Pillar | What it measures | Inputs |
|---|---|---|
| **Q** Quality | Is this a good business, *versus its sector*? | ROIC, FCF yield, gross margin, net debt/EBITDA, earnings stability |
| **C** Confirmation | Is the market agreeing yet? | Vol-adjusted excess return vs SPY at 1/3/6/12m, trend vs 50/200-day |
| **R** Risk | Can you own it, and at what price? | Dollar liquidity, valuation (earnings & EBITDA yield), realised volatility |

Every input is a **cross-sectional percentile**, not a hand-picked threshold. Quality and
valuation rank *within GICS sector*; momentum and liquidity rank across the universe.

**Sector-specific quality profiles.** Banks do not tag operating income or gross profit at
all, so the default quality inputs resolved for a minority of Financials and the rest were
imputed — which measurably compressed the sector (2.84 imputed inputs per name, quality
dispersion 0.090 against 0.167 for Materials; correlation between the two across sectors,
−0.67). Financials now score on ROE, capital, cash-flow yield and earnings stability; Real
Estate on FFO yield, ROE and debt/assets. Financials' quality dispersion rose to 0.161 and
its data confidence from 72% to 98%, while all nine other sectors changed by exactly 0.000.

Regulatory **CET1 is not obtainable** and is not faked: banks tag capital ratios
dimensionally and the bulk XBRL API returns zero filers for it. Equity/assets stands in,
labelled a proxy. FFO is likewise an approximation (net income to common + D&A) and is
labelled as one.

**Factor-level history.** Every night the full decomposition of every name — all fifteen
percentile inputs, the three pillars, confidence — is committed to `ledger/snapshots/`,
each stamped with the hash of the specification that produced it. That is the dataset
Information Coefficient and decile-spread analysis will run against; none of it can be
reconstructed after the fact.

## Why it works this way

This is a rebuild. The previous version published `conviction 0 · AVOID` for all 48 names
for weeks while its workflow reported success. The causes are worth stating, because each
one shaped a design decision here:

| Failure | Design response |
|---|---|
| Fundamentals came from a free API tier capped near 25 requests/day, so 39 of 48 names silently got nothing | Every source is **keyless and quota-free**. SEC's `frames` API returns one concept for *all* filers per request, so fundamentals cost ~80 requests regardless of universe size. |
| ROIC and FCF yield read fields the vendor never returned — structurally always `0` | Fundamentals are computed from **raw XBRL facts**, each carrying the accession number it came from and surfaced in the UI. |
| Daily change was read from a quote blob that didn't contain it, so relative strength was identically `0` for every name | All price features are **derived locally from one OHLCV series**. There is no field to be silently absent. |
| Market caps were multiplied by 1e6 (one name published at `4.4e18`) | `scripts/validate_ledger.py` **fails the build** on implausible values. |
| A quality floor mapped any name with missing inputs to `Q = 0` — absent data read as a confident sell | Missing inputs shrink to the sector median and lower a reported **data confidence**, shown per name. |
| Three of five score multiplied to put a median company at 12/100, so `BUY` was arithmetically unreachable | Geometric mean, which preserves the "a broken pillar vetoes the score" property while placing a median name near 58. |
| Nothing inspected the output | The validator gates every deploy, and the terminal ships a **Data quality** tab. |

## Data sources

| What | Source | Key required |
|---|---|---|
| Fundamentals | [SEC EDGAR XBRL `frames`](https://www.sec.gov/edgar/sec-api-documentation) | No — but SEC requires a contact email in the User-Agent (see below) |
| Prices (daily OHLCV) | Nasdaq, with Yahoo and a committed cache as fallbacks | No |
| Macro | FRED CSV (10y, 2s10s, VIX, HY OAS) | No |
| Universe & sectors | An index-constituent CSV in `data/` | No |

### `SEC_CONTACT` is required

SEC's fair-access policy rejects requests without a reachable contact address — a URL-only
User-Agent and a `@users.noreply.github.com` address both return `403`. Set it to an
address you monitor:

```bash
export SEC_CONTACT='you@example.com'
```

In CI, add it as a repository secret of the same name (`Settings → Secrets and variables →
Actions`). It is the only secret this project needs.

### Universe

Drop an index-holdings export at `data/*.csv` — a raw iShares IWB (Russell 1000) or IVV
file works as-is, preamble rows included. The loader reads ticker, name, GICS sector, index
weight and P/E, and tolerates the usual vendor layout differences. Without one, a committed
~110-name large-cap list is used so the repo always builds.

ETFs in the universe are tracked as **benchmarks only** and never conviction-scored: an
index fund has no ROIC, and scoring one is a category error rather than a data gap.

## Running it

```bash
pip install -e ".[dev]"
export SEC_CONTACT='you@example.com'

python -m equity_monitor.nightly --limit 40      # build ledger/index.json
python scripts/validate_ledger.py                # anti-degeneracy gate
pytest -q                                        # model, parity, regressions

python -m http.server 8000                       # then open /web/terminal.html
```

Useful flags: `--limit N` caps the equity count, `--skip-fundamentals` isolates a price
problem, `--skip-macro` skips FRED.

## Layout

```
equity_monitor/
  universe.py          index constituents -> tickers, sectors, weights
  features.py          prices + filings -> one raw feature dict per name
  model.py             v3 scoring: prepare() ranks, score() is pure
  snapshots.py         nightly factor-level history + score-change attribution
  monitor.py           self-grading: stability, coverage trend, regime, health
  churn.py             why the board moved: information vs. model sensitivity
  swing.py             the 5-25 session board: reward-to-risk gated mean reversion
  watchlist.py         the overnight diff: what changed and whether it matters
  rebuild.py           reconstruct an earlier board from cached prices (never committed)
  nightly.py           orchestrator; writes ledger/
  sources/
    _http.py           retries, throttling, gzip, on-disk cache
    edgar.py           SEC XBRL frames -> fundamentals
    prices.py          OHLCV chain + all price-derived features
    macro.py           FRED series
scripts/validate_ledger.py   CI gate — degenerate ledger, and snapshot-history integrity
scripts/research_swing.py    event study for the swing layer, with its bias controls
web/terminal.html            the terminal (self-contained, no CDN)
tests/                       model behaviour, v2 regressions, JS<->Python parity
```

`ledger/index.json` is the published artifact; `ledger/history/` holds per-symbol OHLCV and
`ledger/history.json` a bundled downsample for the grid.

## Self-monitoring

The validator blocks a bad ledger from shipping. `equity_monitor/monitor.py` grades the
ledgers that did ship, writing `ledger/monitor.json` and driving the terminal's **Monitor**
tab: pass/warn/fail health checks, ranking stability between consecutive snapshots, a tier
migration matrix, a coverage trend, the observed macro regime, and which specification hash
produced each stretch of history.

It measures whether the machinery is working, never whether the scores are right. A board
can be fresh, dispersed, fully covered and perfectly stable while predicting nothing at all.
Whether high-conviction names outperform is an Information Coefficient question that needs
months of accumulated snapshots within a single specification hash — it is not measured
here, and every panel that depends on history states its sample size and what it is still
waiting for rather than rendering a default.

## The swing board

Conviction answers "is this a good business I can own". `equity_monitor/swing.py` answers
a shorter question over 5 to 25 sessions: **this name has fallen, is the bounce worth
taking, and what does it pay against what it risks.** It is a separate board on the same
names, and three of its choices are deliberate departures from how the strategy is usually
written down.

**Reward-to-risk is the gate, not drawdown depth.** A fixed "-15% off the high" entry
paired with a 3xATR stop loses by construction rather than by luck. Volatility scales with
the fall, so the stop widens roughly in step with it, while a fixed-fraction retracement
target grows faster than linearly. Measured across the 1016-name board:

| Below the 52-week high | 38.2% retracement | 3xATR stop | Reward:risk | Share clearing 1.5 |
|---|---|---|---|---|
| 10–15% | +5.3% | −9.8% | **0.53** | 1% |
| 15–20% | +7.9% | −10.5% | **0.75** | 1% |
| 20–25% | +10.9% | −11.9% | 0.93 | 10% |
| 25–30% | +14.2% | −13.5% | 1.07 | 11% |
| 30–40% | +19.9% | −14.9% | 1.35 | 35% |
| 40%+ | +35.0% | −17.9% | 2.35 | 78% |

At the threshold the strategy is usually written with, the trade pays 0.75 for every 1.00
risked and needs a 57% strike rate merely to break even. So the depth threshold is an
*output* of requiring a reward-to-risk floor, not an input to be guessed.

The same arithmetic constrains the stop. Across every band, `3 x ATR(14)` lands within a
point or two of the name's own 25-day standard deviation — 9.8% against 10.0% at the
shallow end, 17.9% against 20.9% at the deep end. A stop at one sigma over the holding
period is inside the noise: roughly **30% of positions are stopped out by drift-free
random walk alone**, before the thesis is right or wrong about anything. Widening the stop
without deepening the entry only moves the loss around.

**Drawdown is measured against the benchmark as well as against the high.** A name 25% off
its high while the index is 20% off its own is beta, and it mean-reverts with the market or
not at all. `rel_drawdown_52w` is the drawdown of the *stock/benchmark ratio*, which
isolates the idiosyncratic part of the fall. It is a ratio of two observed series and needs
no estimated beta — there is no regression to degrade silently on a short history.

**The retracement leg is anchored to the high that started it.** The swing low is the
lowest low *since* the 52-week high, not `lo_52w`, which can predate the high entirely.
Anchoring to it describes a decline that never happened, and every Fibonacci level derived
from it is a number about nothing. `tests/test_swing.py` pins this case.

The 50-day moving average is reported and is deliberately **not** used as a target. It is a
10-week reference being asked to price a 52-week decline: on the current board the median
name 20–25% off its high sits *above* its own 50-day average, so the "target" is behind the
price, and across every band the median touch is worth 0.4% to 5.3%.

One consequence for the existing model. `model.score()` already applies a mean-reversion
uplift to the confirmation pillar for quality names in drawdown, worth a median 1.7
conviction points and up to 5.2. Three names currently sit above the BUY line only because
of it. A swing scanner gated on *conviction* would therefore count the drawdown twice —
once inside the score, once in its own filter. The gate here is the **quality percentile**,
which carries no drawdown term.

The gate is written as a percentile of the score distribution, not as a raw value, because
`q_raw` is a weighted mean of five percentiles and a mean of percentiles is not itself
uniform. On the current board the 65th percentile of `q_raw` is 0.563, while the literal cut
`q_raw >= 0.65` keeps 16% of names rather than 35%.

Every rejected name is counted at the **first gate it failed**, and the funnel is published
alongside the candidates. "Eight candidates tonight" invites the reader to assume the other
thousand were unsuitable businesses; "427 never fell far enough, 379 failed quality, 143
cleared both and failed reward-to-risk" says which. The terminal's **Swing dips** tab draws
the board as quality against drawdown, shaded from the reward-to-risk floor rightward.

### Validating it

`scripts/research_swing.py` runs the event study, against the per-symbol OHLCV
`ledger/history/` already holds. Four things separate it from the study this strategy is
usually validated with, each because the naive version reports a number that cannot be
traded:

* **The exit is simulated along the path, not sampled at the horizon.** Reading
  `close[t+20]` credits the strategy with recoveries a stopped-out position never saw. Every
  event walks forward bar by bar; where a bar touches both stop and target the **stop is
  assumed to fill first**, because a daily bar carries no intrabar sequence and the other
  assumption pays for a coin flip that was not won.
* **Returns are stated against the benchmark over the identical window**, and against a
  **matched control arm** — same name, same exit rule, a random date. Signal minus control
  is the number worth reading; a positive average in a rising market is drift.
* **The interval is a block bootstrap over dates.** A hundred names triggering on one
  afternoon are one observation about one afternoon. Resampling events independently gives
  an interval several times too tight, which is the most common way a dip-buying backtest
  claims 4,000 observations when it has about twenty. `events` and `distinct dates` are
  printed side by side.
* **Two biases are stated rather than removed, because this data cannot remove them.**
  *Survivorship*: the universe is today's constituents, so a name that fell and kept falling
  until it was acquired, delisted or dropped from the index never enters the sample — and
  dip-buying is the strategy most distorted by that, because the excluded names are excluded
  for precisely the outcome being measured. *Quality look-ahead*: fundamentals are as-filed
  today, the same leak `rebuild.py` refuses to write into `ledger/snapshots/`. The
  quality-on and quality-off arms bound it; their gap is edge **plus** leak, so it reads as
  a ceiling. An inverted-quality arm runs as falsification.

Both biases inflate dip-buying, so every number it prints is an upper bound. That is a
reason to hold sizing until `ledger/snapshots/` is deep enough to answer the question
without a leak — the same standard `edge.py` already holds itself to — not a reason to skip
the study.

### Diagnosing churn

Stability says how much the ranking moved. `equity_monitor/churn.py` says why, which is
what decides the response. Both sides are measured in percentile points — how far the
inputs moved, against how far each name's standing in the ranking moved — so the ratio
between them is meaningful. Alongside it: per-input attribution by counterfactual
re-scoring (advance one input to tonight's values, hold the rest at last night's,
re-score under the frozen spec), a count of tier changes caused by moves too small to
mean anything, and per-name elasticity reported with the weakest pillar so the mechanism
is nameable.

It reports evidence, never a prescription. One night is a single reading.

### The overnight diff

`equity_monitor/watchlist.py` builds the morning view: upgrades, downgrades, large moves
that stayed inside a tier, and — kept deliberately separate — tier changes caused by a
move too small to mean anything. That separation is the whole design. The churn
diagnostic measured 39 of 39 tier changes on an ordinary night coming from moves of two
points or less; a "new BUY" list built from those would be wrong in detail every day.
Each row names the largest factor pushing in the direction of the move, and the largest
one pushing against it.

### Hover explanations

Every factor and pillar explains itself on hover — the pillar it feeds and its weight,
the arithmetic, and what the measure is for — from a single registry read by every
surface. Tests assert that every scored input has an entry and that no entry describes an
input the model does not read, so a new factor cannot ship undocumented.

### Reconstructing an earlier board

The snapshot series accumulates one observation per trading day and cannot be back-filled,
so for its first weeks there is nothing to diff against. `equity_monitor/rebuild.py` answers
the same questions from the price history already on disk:

```bash
python -m equity_monitor.rebuild --back 5            # board as of 5 sessions ago, diffed against today
python -m equity_monitor.rebuild --back 20 --site    # …and a browsable terminal pointed at it
```

Price inputs are genuinely reconstructed — relative strength, trend, dollar liquidity and
volatility recomputed from truncated OHLCV and re-ranked among the names that existed then.
The **fundamentals are today's**, because filings are not stored as-of-date. That is
look-ahead bias on the quality pillar, and it is printed on every output.

Which is why the tool **refuses to write into `ledger/snapshots/`**. A reconstructed
snapshot is structurally identical to a recorded one, so a mixed series cannot be un-mixed,
and an Information Coefficient computed over it would be measuring the leak rather than the
model. The refusal is a `realpath` check with its own exit code, not a convention.

### Reading the terminal

The screener ledger pins rank and ticker, rules every fifth row rather than every row, and
groups its eighteen columns with five vertical dividers. The factor map bins 1014 names
into a hex density field and draws the top 40 as points, because at that count a
square-root area scale puts a 25.5px circle on a 786px plot and opacity does not save it;
every hex is hoverable so nothing becomes unreachable. A **position-sizing sandbox** sizes
a book from whatever the screener is showing, and a **factor exposure** panel reports the
book's weighted percentile against the universe median for all fifteen model inputs,
counting only the names whose sector profile actually reads each one.

Baselines are chosen from the data's range rather than defaulted to zero. Sector medians
occupy 46–62 of a hundred-point score, so bars grown from zero differed in length by under
a quarter and all eleven sectors read alike; they now diverge from the universe median,
which is drawn and labelled. The factor-exposure panel keeps its fixed ±50 axis for the
opposite reason — that is the entire range a deviation from a median percentile can take,
and rescaling it to the book would draw a 6-point tilt the same size as a 30-point one.

Charts size to the panel they are in rather than to a hardcoded design width, so a 1540px
card no longer renders a 700px chart against 840px of nothing; the coordinate system stays
1:1 with CSS pixels at every width, so a 9.5px tick label is never scaled up to 28px.

Contrast was measured, not assumed: the small-caps label colour was 3.50:1 on the light
surface, below AA for 10px uppercase text, and is now 5.27:1 light / 7.23:1 dark.

## The parity gate

The terminal recomputes conviction in the browser, so the scoring function exists in both
Python and JavaScript. `tests/test_parity.py` extracts the JS port directly out of
`web/terminal.html`, runs it in node, and asserts agreement across 400 randomised inputs
plus edge cases. It has already caught one real divergence: Python's `round()` is banker's
rounding and JavaScript's `Math.round()` rounds half away from zero, so both now use
`floor(x + 0.5)`.

Splitting cross-sectional ranking (Python only) from a pure `score(percentiles)` is what
keeps this tractable — there is exactly one pure function to hold in sync.

## Limitations

- Fundamentals are as-filed and lag the market by up to a quarter. TTM figures roll the last
  fiscal year forward using the most recent reported quarter, and only when all three legs
  share an XBRL tag — mixing tags produces plausible nonsense.
- Dual-class companies (BRK.B) report share counts as dimensional facts the bulk `frames`
  API omits, so market cap can be missing. It is shown missing rather than guessed.
- Net debt/EBITDA and FCF yield are not meaningful for banks and insurers; those names take
  the sector median and a lower confidence score.
- ROIC is suppressed where invested capital is negative or near zero (heavy buybacks), since
  the ratio there has a confident sign and no meaning.
- Percentiles are relative to *this* universe. Conviction 80 means top of this list, not an
  absolute valuation claim.

**This is a research tool, not investment advice.**
