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
  nightly.py           orchestrator; writes ledger/
  sources/
    _http.py           retries, throttling, gzip, on-disk cache
    edgar.py           SEC XBRL frames -> fundamentals
    prices.py          OHLCV chain + all price-derived features
    macro.py           FRED series
scripts/validate_ledger.py   CI gate — fails the build on a degenerate ledger
web/terminal.html            the terminal (self-contained, no CDN)
tests/                       model behaviour, v2 regressions, JS<->Python parity
```

`ledger/index.json` is the published artifact; `ledger/history/` holds per-symbol OHLCV and
`ledger/history.json` a bundled downsample for the grid.

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
