"""Equity Conviction Monitor — quality mean-reversion terminal.

Public surface:
    equity_monitor.model.score(quote, btc_or_spy, sector_bench) -> (quality, conf, risk, conv, sig, comp)
    equity_monitor.cli.main(...) -> nightly refresh + paper-index ledger

The model is the equity analogue of the crypto terminal's v2 multiplicative engine:
    Conviction = 100 * Quality * Confirmation * RiskAdjustment
"""
from .model import score, signal, quality, confirmation, risk_adjustment  # noqa: F401

__all__ = ["score", "signal", "quality", "confirmation", "risk_adjustment"]
__version__ = "0.1.0"
