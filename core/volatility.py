"""
Volatility Engine: ATR and Volatility Percentile.

Fully vectorized, no loops over DataFrame rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. ATR (Average True Range) — vectorized
# ---------------------------------------------------------------------------

def atr(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Compute Average True Range using Welles Wilder smoothing.

    True Range = max(
        High - Low,
        |High - Close_prev|,
        |Low - Close_prev|
    )

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ticker``, ``date``, ``high``, ``low``, ``close``.
    window : int
        Smoothing window.  Default 14.

    Returns
    -------
    pd.DataFrame
        Original frame with added column ``atr_{window}``.
    """
    out = df.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]
    close_prev = out.groupby("ticker", sort=False)["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Welles Wilder: EMA with alpha = 1/window
    alpha = 1.0 / window
    out[f"atr_{window}"] = true_range.groupby(out["ticker"]).transform(
        lambda s: s.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    )
    return out


# ---------------------------------------------------------------------------
# 2. Volatility Percentile
# ---------------------------------------------------------------------------

def volatility_percentile(
    df: pd.DataFrame,
    atr_window: int = 14,
    lookback: int = 252,
) -> pd.DataFrame:
    """Compute where current ATR ranks within its own rolling history.

    A value of 0.90 means current ATR is higher than 90% of the last
    *lookback* observations — a high-volatility regime.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ticker``, ``date``, ``high``, ``low``, ``close``.
        Should already have ATR computed (this function computes it if
        missing).
    atr_window : int
        ATR smoothing window.
    lookback : int
        Rolling window for percentile ranking.  Default 252 trading days.

    Returns
    -------
    pd.DataFrame
        Original frame with added column ``vol_percentile``.
    """
    atr_col = f"atr_{atr_window}"
    if atr_col not in df.columns:
        out = atr(df, window=atr_window)
    else:
        out = df.copy()

    grp = out.groupby("ticker", sort=False)[atr_col]

    # Rank current ATR within rolling window
    def _rolling_pctile(s: pd.Series) -> pd.Series:
        return s.rolling(lookback, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    out["vol_percentile"] = grp.transform(_rolling_pctile)
    return out


# ---------------------------------------------------------------------------
# 3. Dynamic Z-score threshold adjustment
# ---------------------------------------------------------------------------

def dynamic_zscore_threshold(
    base_threshold: float = 2.0,
    vol_percentile: pd.Series | None = None,
) -> pd.Series:
    """Widen Z-score thresholds in high-vol regimes, tighten in low-vol.

    Adjustment rule:
        adjusted = base * (1 + 0.5 * (vol_pctile - 0.5))

    At vol_percentile = 0.0  →  threshold = base * 0.75  (tighter)
    At vol_percentile = 0.5  →  threshold = base * 1.0   (unchanged)
    At vol_percentile = 1.0  →  threshold = base * 1.25  (wider)

    Parameters
    ----------
    base_threshold : float
    vol_percentile : pd.Series
        Output from ``volatility_percentile()``.  If *None*, returns
        a constant series of *base_threshold*.

    Returns
    -------
    pd.Series
        Adjusted threshold for each row.
    """
    if vol_percentile is None:
        return pd.Series(base_threshold, index=pd.RangeIndex(0))

    factor = 1.0 + 0.5 * (vol_percentile - 0.5)
    return base_threshold * factor


# ---------------------------------------------------------------------------
# 4. Combined volatility signals
# ---------------------------------------------------------------------------

def volatility_signals(
    df: pd.DataFrame,
    atr_window: int = 14,
    vol_lookback: int = 252,
) -> pd.DataFrame:
    """Add ATR and volatility percentile to *df*."""
    out = atr(df, window=atr_window)
    out = volatility_percentile(out, atr_window=atr_window, lookback=vol_lookback)
    return out
