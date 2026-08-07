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
