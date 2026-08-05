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

## Go live (2 steps — both needed, neither is committed)

1. **Flip GitHub Pages on** (no code fix possible here — it's a UI toggle):
   `Settings → Pages → Build and deployment → Source: Deploy from a branch`
   → select `master` branch, `/` (root) folder → Save.
   Once on, the terminal serves at `https://alecmessino.github.io/equity-conviction-monitor/`.

2. **Set FMP_API_KEY** (`Settings → Secrets and variables → Actions`):
   - `FMP_API_KEY` — **required** for live S&P 500 / Russell 1000 data.
     Create a free account at financialmodelingprep.com → Dashboard → API Keys.
     NOTE: the key printed in FMP's docs page is a **placeholder** (returns 401
     `[]` on `/stable/`); generate a real one from your dashboard.
   - `POLYGON_API_KEY` — optional, for higher-fi intraday/Vol surfaces.

**Without FMP_API_KEY**: the terminal still serves and renders (shows the fixture
data + v2 model), but scores are placeholder-calibrated, not market data. No
data is ever fabricated.
