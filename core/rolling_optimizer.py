"""
Rolling Window Optimizer: Finds the best signal parameters for the most recent period.
"""
from __future__ import annotations
import itertools
import logging
from core.price_collector import PriceDataCollector
from core.backtest import run_backtest, BacktestConfig

logger = logging.getLogger(__name__)

def find_optimal_params(ticker: str, lookback_years: int = 3) -> dict:
    """Find the best zscore_threshold for a ticker over the last N years."""
    from datetime import datetime
    end_year = datetime.now().year
    start_year = end_year - lookback_years
    
    # We'll test a range of Z-score thresholds
    thresholds = [1.5, 2.0, 2.5, 3.0]
    best_sharpe = -999
    best_thresh = 2.0
    
    for thresh in thresholds:
        try:
            res = run_backtest(BacktestConfig(
                tickers=[ticker], 
                start=f"{start_year}-01-01",
                zscore_threshold=thresh
            ))
            if res['sharpe'] > best_sharpe:
                best_sharpe = res['sharpe']
                best_thresh = thresh
        except:
            continue
            
    return {"zscore_threshold": best_thresh, "sharpe": best_sharpe}

if __name__ == "__main__":
    # Quick test on the watchlist
    watchlist = ["AVB", "HST", "LRCX", "NKE", "PPL", "ES", "CMCSA", "ABT"]
    for t in watchlist:
        opt = find_optimal_params(t)
        print(f"{t}: Optimal Z={opt['zscore_threshold']} (Recent Sharpe: {opt['sharpe']:.2f})")
