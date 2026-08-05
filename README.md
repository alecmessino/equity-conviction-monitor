Equity Conviction Monitor
# Equity Conviction Monitor — Quality mean-reversal terminal

> A **distinct terminal** to the crypto conviction monitor. Built around the same
> v2 multiplicative philosophy — `Conviction = 100 × Quality × Confirmation × RiskAdj` —
> but with equity-native factors: fundamentals, multi-horizon relative strength vs
> sector/SPX, mean-reversion optics, and factor/ETF relative-value overlays.

## Scope

- **Universe**: S&P 500 + Russell 1000 constituents + major regional/factor ETFs
  (`EWJ, EFA, VEU, IWD, QUAL, VLUE, MTUM, USMV, IWM, EEM, VWO, AAXJ` and the rest).
- **Data**: Polygon / FMP primary (price, volume, ADV); FMP / Tiingo for
  fundamentals; FRED + ETF.com for factor ETFs. 15-min intraday refresh.
- **Model**:
  - **Quality** (0–1): ROIC, gross-margin stability, FCF yield, debt/EBITDA,
    5-yr earnings-consistency. Mega-cap blue-chip floor.
  - **Confirmation**: log-return RS vs SPX + sector, vol-normalised, soft tanh
    (NO hard clamp → rankable within a tier). 3–12 month drawdown from 52wk.
  - **RiskAdj**: liquidity (ADV/mc), short interest / days-to-cover, valuation
    z-score (forward P/E or EV/EBITDA vs 5-yr median).

## Modules

- `A` Liquidity & Short-Interest
- `B` Multi-horizon RS + drawdown confirmation
- `C` Fundamentals (Quality composite)
- `D` Factor Attribution (Value / Quality / Momentum / Size / Low-Vol)
- `E` Mean-Reversion Overlay (RSI14 troughs, Bollinger %B, 200d distance)
- Paper index of top-10 conviction names, score-proportional weights.

## Pipeline (mirrors crypto repo)

```
Data (FMP/Polygon) -> Features -> v2 Multiplicative Model -> Portfolio -> Ledger -> UI
```

Every score is byte-identical between the browser (vanilla JS) and the nightly
builder (Python) and is gated by `tests/test_parity.py` (frontend/backend parity)
and the FROZEN_CONVICTION regression fixture.

## Go live (2 steps — both needed)

1. **Enable Pages with the Actions source** (one-time UI toggle):
   `Settings → Pages → Build and deployment → Source: GitHub Actions`
   (do NOT pick "Deploy from a branch" — this repo uses the `pages.yml` workflow).
   The terminal then serves at `https://alecmessino.github.io/equity-conviction-monitor/web/terminal.html`.

2. **Set `FMP_API_KEY`** (`Settings → Secrets and variables → Actions → New repository secret`):
   - `FMP_API_KEY` — **required** for live market data. Free tier covers
     `quote`, `profile`, `income-statement`, `ratios`, `balance-sheet-statement`,
     `cash-flow-statement` (everything the model consumes).
     Create a free account at financialmodelingprep.com → Dashboard → API Keys.
     The key printed in FMP's docs page is a **placeholder** (returns `404 []` on
     `/stable/`); generate a real one from your dashboard.
   - `POLYGON_API_KEY` — optional (higher-fi intraday/Vol surfaces).

**After setting the key**, trigger the data refresh: `Actions → pages.yml → Run workflow`
(or push to `master`). The nightly (`equity_monitor/nightly.py`) pulls fresh features,
scores them, and rewrites `ledger/index.json` — the terminal then shows live prices
and **model-computed** convictions, not the committed fixture.

Without `FMP_API_KEY`: the terminal still serves and renders, but `ledger/index.json`
is the committed fixture (model-computed sample, clearly labelled). No data is ever fabricated.
