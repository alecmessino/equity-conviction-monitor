"""Equity Conviction Monitor — quality mean-reversion terminal."""
from .model import score, prepare, score_rows, signal, WEIGHTS  # noqa: F401

__all__ = ["score", "prepare", "score_rows", "signal", "WEIGHTS"]
__version__ = "0.3.0"
