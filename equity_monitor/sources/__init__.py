"""External data sources.

Every source in this package is KEYLESS. That is a deliberate architectural
constraint, not a limitation we settled for: the previous iteration depended on
Alpha Vantage's free tier, whose ~25 requests/day ceiling silently truncated the
universe after 9 of 48 symbols and left the remaining 39 with null fundamentals.
A pipeline that degrades invisibly when a quota is hit is worse than no pipeline.

  * ``prices``  — daily OHLCV (Nasdaq primary, Yahoo fallback, committed cache last)
  * ``edgar``   — SEC XBRL company facts via the bulk ``frames`` API
  * ``macro``   — FRED daily series (CSV download endpoint, no key)
"""
