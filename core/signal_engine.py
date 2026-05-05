"""
Signal Engine: Unified aggregator combining mean-reversion, momentum, and
volatility modules into a single signal DataFrame.

Usage
-----
    from price_collector import PriceDataCollector
    from signal_engine import generate_signals

    df = PriceDataCollector(tickers=["AAPL", "TSLA"]).collect()
    signals = generate_signals(df)
    print(signals[signals["signal"] != "NEUTRAL"])
"""

from __future__ import annotations

import logging

import pandas as pd

from core.mean_reversion import mean_reversion_signals
from core.momentum import momentum_signals
from core.volatility import volatility_signals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal aggregation logic
# ---------------------------------------------------------------------------

def _combine_signals(
    df: pd.DataFrame,
    rsi_oversold: float = 25.0,
    rsi_overbought: float = 75.0,
    base_zscore_threshold: float = 2.0,
) -> pd.DataFrame:
    """Merge sub-signals into a single ``signal`` column.

    Priority order:
    1. Momentum override: EXHAUSTED_DOWN + oversold → potential LONG bounce
    2. Mean-reversion LONG/SHORT (confirmed by momentum not being STRONG_DOWN/UP)
    3. Default: NEUTRAL
    """
    out = df.copy()
    out["signal"] = "NEUTRAL"

    z_kf = out["ou_kf_zscore"]
    mr_sig = out["mr_signal"]        # LONG / SHORT / NEUTRAL
    regime = out["momentum_regime"]  # STRONG_UP, EXHAUSTED_DOWN, ...
    rsi_col = out[f"rsi_14"]
    roc_20 = out[f"roc_20"]
    vol_pct = out["vol_percentile"] if "vol_percentile" in out.columns else pd.Series(0.5, index=out.index)

    # --- Dynamic Z-Score Threshold ---
    # Widen thresholds in high volatility (vol_pct > 0.75) to avoid noise.
    # Tighten thresholds in low volatility (vol_pct < 0.25) to catch moves.
    # Using a conservative 0.25 multiplier to prevent excessive noise.
    dynamic_threshold = base_zscore_threshold * (1.0 + 0.25 * (vol_pct - 0.5))
    
    # --- Regime Filter: Don't catch falling knives in strong trends ---
    # Stricter filters to prevent "value traps" in structural declines/rallies.
    out["regime_safe_long"] = roc_20 > -0.05
    out["regime_safe_short"] = roc_20 < 0.05

    # --- Mean-reversion signals with confirmation ---
    # Use dynamic_threshold instead of fixed base_zscore_threshold
    long_mask = (z_kf < -dynamic_threshold) & out["regime_safe_long"] & (regime != "STRONG_DOWN")
    out.loc[long_mask, "signal"] = "LONG_REVERSION"

    short_mask = (z_kf > dynamic_threshold) & out["regime_safe_short"] & (regime != "STRONG_UP")
    out.loc[short_mask, "signal"] = "SHORT_REVERSION"

    # --- Exhaustion bounces (secondary signal) ---
    exhausted_long = (regime == "EXHAUSTED_DOWN") & (rsi_col < rsi_oversold)
    out.loc[exhausted_long & (out["signal"] == "NEUTRAL"), "signal"] = "LONG_BOUNCE"

    exhausted_short = (regime == "EXHAUSTED_UP") & (rsi_col > rsi_overbought)
    out.loc[exhausted_short & (out["signal"] == "NEUTRAL"), "signal"] = "SHORT_BOUNCE"

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    zscore_threshold: float = 2.0,
    rsi_oversold: float = 25.0,
    rsi_overbought: float = 75.0,
) -> pd.DataFrame:
    """Run the full Phase B signal pipeline on raw price data.

    Parameters
    ----------
    df : pd.DataFrame
        Output from ``price_collector.PriceDataCollector.collect()``.
    zscore_threshold : float
        Base threshold for OU-KF Z-score. Will be adjusted dynamically by volatility.
    rsi_oversold : float
        RSI level considered oversold (for bounce signals).
    rsi_overbought : float
        RSI level considered overbought (for short bounce signals).

    Returns
    -------
    pd.DataFrame
        Augmented frame with signal columns.
    """
    logger.info("Computing mean-reversion signals (rolling + OU-KF) …")
    # Pass a very low threshold here so we don't filter out signals before dynamic adjustment
    out = mean_reversion_signals(df, price_col="close", zscore_threshold=0.5)

    logger.info("Computing momentum signals (ROC + RSI) …")
    out = momentum_signals(out, price_col="close")

    logger.info("Computing volatility signals (ATR + vol percentile) …")
    out = volatility_signals(out, atr_window=14, vol_lookback=252)

    logger.info("Aggregating signals with dynamic thresholds …")
    
    # Apply dynamic thresholding logic
    out = _combine_signals(
        out, 
        rsi_oversold=rsi_oversold, 
        rsi_overbought=rsi_overbought,
        base_zscore_threshold=zscore_threshold
    )

    logger.info("Computing dynamic thresholds …")
    vol_pct = out["vol_percentile"] if "vol_percentile" in out.columns else pd.Series(0.5, index=out.index)
    out["dynamic_zscore_threshold"] = zscore_threshold * (1.0 + 0.25 * (vol_pct - 0.5))

    # Summary
    sig_counts = out["signal"].value_counts()
    logger.info("Signal distribution:\n%s", sig_counts.to_string())

    return out


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from price_collector import PriceDataCollector

    collector = PriceDataCollector(
        tickers=["VOO"],
        start="2024-01-01",
    )
    prices = collector.collect()
    signals = generate_signals(prices)

    # Show only non-neutral signals
    active = signals[signals["signal"] != "NEUTRAL"]
    print(f"\nActive signals: {len(active)} / {len(signals)}")
    if not active.empty:
        print(active[["date", "ticker", "close", "signal", "ou_kf_zscore", "rsi_14", "momentum_regime"]].tail(20).to_string(index=False))
