"""Equity Conviction Monitor — quality mean-reversion terminal."""
from .model import score, signal, quality, confirmation, risk_adjustment  # noqa: F401
from . import data, rs, nightly  # noqa: F401

__all__ = ["score", "signal", "quality", "confirmation", "risk_adjustment",
           "data", "rs", "nightly"]
__version__ = "0.2.0"
