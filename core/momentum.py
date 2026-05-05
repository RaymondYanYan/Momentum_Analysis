"""
Momentum Filter: Rate of Change (ROC) and RSI.

Fully vectorized, no loops over DataFrame rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Rate of Change (ROC)
# ---------------------------------------------------------------------------

def rate_of_change(
    df: pd.DataFrame,
    price_col: str = "close",
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """Compute ROC for multiple lookback periods.

    ROC_t = (P_t / P_{t-n}) - 1

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ticker``, ``date``, and ``price_col``.
    price_col : str
    periods : list[int]
        Lookback periods.  Defaults to ``[5, 10, 20]``.

    Returns
    -------
    pd.DataFrame
        Original frame with added columns ``roc_{n}`` for each period *n*.
    """
    if periods is None:
        periods = [5, 10, 20]

    out = df.copy()
    for n in periods:
        grp = out.groupby("ticker", sort=False)[price_col]
        price_lag = grp.transform(lambda s: s.shift(n))
        out[f"roc_{n}"] = (out[price_col] / price_lag) - 1.0
    return out


# ---------------------------------------------------------------------------
# 2. RSI (Relative Strength Index) — vectorized Welles Wilder smoothing
# ---------------------------------------------------------------------------

def rsi(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 14,
) -> pd.DataFrame:
    """Compute RSI using vectorized Welles Wilder smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ticker``, ``date``, and ``price_col``.
    price_col : str
    window : int
        RSI lookback window.  Default 14.

    Returns
    -------
    pd.DataFrame
        Original frame with added column ``rsi_{window}``.
    """
    out = df.copy()
    delta = out.groupby("ticker", sort=False)[price_col].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Welles Wilder smoothing = exponential moving average with alpha=1/window
    alpha = 1.0 / window
    avg_gain = gain.groupby(out["ticker"]).transform(
        lambda s: s.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    )
    avg_loss = loss.groupby(out["ticker"]).transform(
        lambda s: s.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    )

    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    out[f"rsi_{window}"] = 100.0 - (100.0 / (1.0 + rs))
    return out


# ---------------------------------------------------------------------------
# 3. Momentum regime classification
# ---------------------------------------------------------------------------

def momentum_regime(
    df: pd.DataFrame,
    roc_period: int = 10,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """Classify each row into a momentum regime.

    Regimes
    -------
    STRONG_UP    : ROC > 0 and RSI > 60
    STRONG_DOWN  : ROC < 0 and RSI < 40
    EXHAUSTED_UP : ROC > 0 and RSI > 75  (overbought, momentum may fade)
    EXHAUSTED_DOWN: ROC < 0 and RSI < 25  (oversold, bounce likely)
    NEUTRAL      : everything else
    """
    out = rate_of_change(df, periods=[roc_period])
    out = rsi(out, window=rsi_window)

    roc = out[f"roc_{roc_period}"]
    rsi_col = out[f"rsi_{rsi_window}"]

    regime = pd.Series("NEUTRAL", index=out.index)
    regime[(roc > 0) & (rsi_col > 60) & (rsi_col <= 75)] = "STRONG_UP"
    regime[(roc < 0) & (rsi_col < 40) & (rsi_col >= 25)] = "STRONG_DOWN"
    regime[(roc > 0) & (rsi_col > 75)] = "EXHAUSTED_UP"
    regime[(roc < 0) & (rsi_col < 25)] = "EXHAUSTED_DOWN"

    out["momentum_regime"] = regime
    return out


# ---------------------------------------------------------------------------
# 4. Combined momentum filter
# ---------------------------------------------------------------------------

def momentum_signals(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add momentum indicators and regime labels to *df*."""
    out = rate_of_change(df, price_col=price_col, periods=[5, 10, 20])
    out = rsi(out, price_col=price_col, window=14)
    out = momentum_regime(out, roc_period=10, rsi_window=14)
    return out
