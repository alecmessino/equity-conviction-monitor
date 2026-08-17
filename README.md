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
  watchlist.py         the overnight diff: what changed and whether it matters
  rebuild.py           reconstruct an earlier board from cached prices (never committed)
  nightly.py           orchestrator; writes ledger/
  sources/
    _http.py           retries, throttling, gzip, on-disk cache
    edgar.py           SEC XBRL frames -> fundamentals
    prices.py          OHLCV chain + all price-derived features
    macro.py           FRED series
scripts/validate_ledger.py   CI gate — degenerate ledger, and snapshot-history integrity
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

### Driving the terminal

The terminal is addressed by **function code**, not only by clicking. `⌘K` / `Ctrl+K`
opens the command bar; every code takes its argument inline, and anything that matches no
code at all falls through to a fuzzy search over tickers and company names — including
transpositions, so a mistyped code lands on the function you meant rather than on an error.

| | | | |
|---|---|---|---|
| `DES <ticker>` tear sheet | `SCR [sector\|signal\|text]` screener | `RV <ticker>` sector peers | `PORT` book and sizing |
| `WL` overnight diff | `MON` pipeline health | `SHK [pct]` price scenario | `TOP [n]` top N |
| `FKQ [factor]` factor leaders | `CSV` export | `SEC` · `DQ` · `MTH` views | `HELP` list them all |

Keyboard: `1`–`8` jump views, `J`/`K` walk the ledger, `g`/`G` and `PgUp`/`PgDn` move in
bulk, `Enter` opens the row under the cursor, `R` peers, `S` scenario, `M` collapses the
overview, `E` exports, `T` theme. `Esc` is a hierarchy rather than one action — command
bar, then detail panel, then a pill or `TOP` selection, then the filters, then the cursor —
because at any moment several things are open and dismissing the wrong one loses work.

A **market & book overview** sits above all eight views — one block, not a row of cards,
because everything below it is name-level and a desk that opens on a ticker has skipped
the two questions that decide whether the ticker matters. It carries Morningstar-style
return boxes where the cell *is* the swatch, tinted by magnitude and capped at 37% alpha
so the figure stays `--ink` at every step: index and region across 1D/1M/3M/YTD/1Y, the
book against the benchmark, and the style ETFs as excess in percentage points (excess
rather than total — these are all US large-cap funds, so their totals agree to a point or
two and a total-return column would read as five copies of the index). Under it, a
normalised chart rebased to **0%** at the window start with up to six selectable legs and
a hover that reads every line at one date, the FRED rates & credit column, and the eleven
sector SPDRs ordered by the selected horizon. Universe breadth and the model's own factor
tilt sit with the board on the Screener, since they describe the scored names rather than
the tape.

Growth is proxied by the Nasdaq-100 and labelled `†`: there is no growth ETF in the
committed universe, and substituting one that is not there would be the kind of quiet
invention this rebuild exists to remove. Tracking error and the systematic share of
variance do not exist upstream either — there is no covariance matrix in the ledger — so
they are computed in the browser from the published weights against the bundled series
and say so wherever they appear.

The palette is teal and coral over a warm neutral, light mode primary. Measured on the
light surface: teal 6.06:1, coral 4.82:1. Teal/coral separates on the blue-yellow axis as
well as red-green, so it survives protanopia and deuteranopia where the previous blue/red
pair narrowed sharply — and the whole system moves, not only the diverging poles, because
leaving the sequential ramp blue would put "high on this factor" and "above the median" in
the same hue on the same screen.

The ledger is **virtualised** — "Show all" is 1014 rows across 19 columns, and building
all of them costs ~19,000 nodes on every keystroke in the search box. Only the visible
window plus an overscan margin reaches the DOM; two spacer rows carry the rest of the
height so the scrollbar still describes the real list. The row pitch lives in one CSS
custom property that the virtualiser reads back, because a JS constant and a CSS rule that
disagree by a pixel put the spacers out by a row every forty rows.

Position sizing solves its three ceilings **together**. A per-position cap, a per-sector
cap and a per-name liquidity cap set by ADV and participation cannot be applied in
sequence: capping positions, redistributing, then capping sectors pushes weight back over
the position ceiling, and only the last pass applied is actually satisfied. A convergent
water-fill fills the free names in proportion to conviction, advances only as far as the
first ceiling it would cross, freezes it and refills the remainder — bounded at
names + sectors rounds. Whatever cannot be placed is **cash, reported as cash**; pushing
refused size into the next name concentrates the book into whatever happened to be liquid
and then presents that as the portfolio the model recommended.

The scenario panel shocks every price and re-scores under the frozen specification. What
moves: drawdown from the 52-week high, position against the 50- and 200-day averages, the
vol-adjusted excess return, and the earnings and EBITDA yields. What does not: ROIC, gross
margin, leverage, earnings stability — a one-day tape does not change what a business
earns. A *uniform* shock therefore barely moves the board, and the panel says so rather
than looking broken: every input is a percentile, and a monotone transform of an input
cannot reorder a rank, so the entire effect runs through the drawdown gate. The dispersion
slider scales each move by the name's own realised volatility, and that is what re-ranks.
Nothing is written anywhere.

The tear sheet fits an OLS channel to the **log** of the close. A line fitted to the level
implies a constant dollar drift, so the same channel is proportionally four times as wide
at $20 as at $80 and the name looks calmer the higher it goes; in logs the drift is a rate
and the band a constant percentage. Drift is annualised only when the bars carry dates —
the bundled `history.json` is a downsample, and multiplying its per-bar drift by 252
reported a 2%-per-week trend as +265%/yr.

### The earnings calendar

There is no keyless forward earnings calendar. Vendors sell one; SEC does not publish
one, because no company is obliged to file its reporting date in a structured form. So
`equity_monitor/earnings.py` does not pretend to have one — it measures what SEC *does*
publish, which is when every filer actually reported, and projects the next date from
that filer's own cadence.

Every row is one of two things, and the distinction is the whole panel:

- **confirmed** — a filing has landed, with the form that reported it beside the date.
- **estimated** — no filing yet. The date is the filer's last period end plus the lag it
  has historically taken to report: 32 days at the universe median over 899 observations,
  ±4 at the median absolute deviation. These are wrong by a few days routinely.

A calendar that renders those alike is worse than no calendar, because a desk will
schedule around the projection exactly as if it were the fact.

The history comes from EDGAR's own index files — one request covers every filer at once,
the same property that makes the `frames` API affordable for fundamentals. Completed
quarters are immutable and cached indefinitely; the current quarter is assembled from
daily index files, each immutable once the day is over. **There is no per-ticker loop**,
and on a warm cache a run fetches exactly one file: today's. Names that cannot be matched
to a CIK carry no date rather than a guessed one.

Two defects worth recording, because both were silent and both produced plausible
numbers. The daily index writes `20260814` where the quarterly writes `2026-05-07`, so a
reader looking for an ISO date found none and matched inside the accession number
instead — yielding four-digit CIKs and dates like `1371-26-00`, with the only symptom
being a calendar containing no filing newer than the last completed quarter. And
`max(end)` over a name's facts picks the dei cover-page date, not the period end: NetApp
files sixteen facts dated 2026-04-24 and one dated 2026-05-28, so pairing the maximum
with the filing measured cover-date-to-filing and reported an 8-day median lag for a form
that takes a month. Every observed lag came out at exactly 8 — a distribution with no
spread is the signature of measuring the wrong quantity, not of a punctual universe. The
period end is now the mode across the name's facts, and the index parser reads columns
positionally rather than hunting for a date.

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
