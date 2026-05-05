"""
Universe Scanner: Run the mean-reversion strategy across all S&P 500 constituents.

Purpose
-------
Identify which sectors and individual tickers the strategy works best on,
so we can focus monitoring and capital allocation on the highest-fit names.

Usage
-----
    ./venv/bin/python universe_scan.py [--start DATE] [--n-workers N]

Output
------
- CSV file: `sp500_scan_results.csv` with per-ticker metrics
- Summary: top/bottom 20 tickers by Sharpe, sector-level aggregates
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from core.backtest import run_backtest, BacktestConfig
from core.price_collector import PriceDataCollector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S&P 500 Constituents (as of 2024-2025)
# Source: Wikipedia / public lists. May drift over time.
# ---------------------------------------------------------------------------

SP500_TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ACN", "CSCO", "ADBE", "CRM", "AMD",
    "INTC", "TXN", "QCOM", "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC", "SNPS",
    "CDNS", "MCHP", "NXPI", "FTNT", "PANW", "WDAY", "ADSK", "TEAM", "DDOG", "CRWD",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN",
    "ISRG", "GILD", "VRTX", "REGN", "ZTS", "BSX", "SYK", "BDX", "EW", "HCA",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SPGI",
    "AXP", "C", "SCHW", "CB", "PGR", "MMC", "ICE", "CME", "AON", "USB",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "CMG",
    "ABNB", "EL", "MAR", "ROST", "YUM", "ORLY", "AZO", "APTV", "HLT", "DHI",
    # Consumer Staples
    "WMT", "PG", "COST", "PEP", "KO", "PM", "MO", "CL", "MDLZ", "GIS",
    "KHC", "KMB", "SYY", "HSY", "STZ", "CLX", "CHD", "TSN", "CAG", "K",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
    "WMB", "KMI", "HES", "DVN", "FANG", "BKR", "EQT", "CTRA", "MRO", "OKE",
    # Industrials
    "UNP", "HON", "UPS", "RTX", "BA", "CAT", "DE", "GE", "LMT", "MMM",
    "GD", "NOC", "EMR", "ETN", "ITW", "CSX", "NSC", "WM", "RSG", "PCAR",
    # Materials
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DOW", "DD", "NUE", "VMC",
    "MLM", "PPG", "ALB", "BALL", "CE", "CF", "MOS", "FMC", "EMN", "IP",
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "PSA", "WELL", "DLR", "O", "SBAC", "AVB",
    "EQR", "INVH", "MAA", "KIM", "REXR", "VTR", "ARE", "ESS", "UDR", "HST",
    # Utilities
    "NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "WEC",
    "ES", "AWK", "DTE", "PPL", "PEG", "EIX", "FE", "ETR", "AEE", "CNP",
    # Communication Services
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR",
    "EA", "ATVI", "TTWO", "OMC", "IPG", "NWSA", "NWS", "FOXA", "FOX", "PARA",
    # Additional / miscellaneous
    "BRK.A", "PYPL", "SQ", "SHOP", "UBER", "LYFT", "COIN", "HOOD", "DKNG", "RBLX",
]

# Remove duplicates and sort
SP500_TICKERS = sorted(set(SP500_TICKERS))

# Sector mapping (simplified — manual assignment for key names)
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AVGO": "Technology",
    "ORCL": "Technology", "ACN": "Technology", "CSCO": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "AMD": "Technology", "INTC": "Technology", "TXN": "Technology",
    "QCOM": "Technology", "INTU": "Technology", "AMAT": "Technology", "MU": "Technology",
    "ADI": "Technology", "LRCX": "Technology", "KLAC": "Technology", "SNPS": "Technology",
    "CDNS": "Technology", "MCHP": "Technology", "NXPI": "Technology", "FTNT": "Technology",
    "PANW": "Technology", "WDAY": "Technology", "ADSK": "Technology", "TEAM": "Technology",
    "DDOG": "Technology", "CRWD": "Technology",
    # Healthcare
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "PFE": "Healthcare", "AMGN": "Healthcare", "ISRG": "Healthcare", "GILD": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare", "ZTS": "Healthcare", "BSX": "Healthcare",
    "SYK": "Healthcare", "BDX": "Healthcare", "EW": "Healthcare", "HCA": "Healthcare",
    # Financials
    "BRK.B": "Financials", "BRK.A": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "BLK": "Financials", "SPGI": "Financials", "AXP": "Financials",
    "C": "Financials", "SCHW": "Financials", "CB": "Financials", "PGR": "Financials",
    "MMC": "Financials", "ICE": "Financials", "CME": "Financials", "AON": "Financials",
    "USB": "Financials",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "LOW": "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "CMG": "Consumer Discretionary", "ABNB": "Consumer Discretionary", "EL": "Consumer Discretionary",
    "MAR": "Consumer Discretionary", "ROST": "Consumer Discretionary", "YUM": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary", "AZO": "Consumer Discretionary", "APTV": "Consumer Discretionary",
    "HLT": "Consumer Discretionary", "DHI": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "COST": "Consumer Staples",
    "PEP": "Consumer Staples", "KO": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples", "MDLZ": "Consumer Staples",
    "GIS": "Consumer Staples", "KHC": "Consumer Staples", "KMB": "Consumer Staples",
    "SYY": "Consumer Staples", "HSY": "Consumer Staples", "STZ": "Consumer Staples",
    "CLX": "Consumer Staples", "CHD": "Consumer Staples", "TSN": "Consumer Staples",
    "CAG": "Consumer Staples", "K": "Consumer Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",
    "OXY": "Energy", "HAL": "Energy", "WMB": "Energy", "KMI": "Energy",
    "HES": "Energy", "DVN": "Energy", "FANG": "Energy", "BKR": "Energy",
    "EQT": "Energy", "CTRA": "Energy", "MRO": "Energy", "OKE": "Energy",
    # Industrials
    "UNP": "Industrials", "HON": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "BA": "Industrials", "CAT": "Industrials", "DE": "Industrials", "GE": "Industrials",
    "LMT": "Industrials", "MMM": "Industrials", "GD": "Industrials", "NOC": "Industrials",
    "EMR": "Industrials", "ETN": "Industrials", "ITW": "Industrials", "CSX": "Industrials",
    "NSC": "Industrials", "WM": "Industrials", "RSG": "Industrials", "PCAR": "Industrials",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials", "DOW": "Materials", "DD": "Materials",
    "NUE": "Materials", "VMC": "Materials", "MLM": "Materials", "PPG": "Materials",
    "ALB": "Materials", "BALL": "Materials", "CE": "Materials", "CF": "Materials",
    "MOS": "Materials", "FMC": "Materials", "EMN": "Materials", "IP": "Materials",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
    "PSA": "Real Estate", "WELL": "Real Estate", "DLR": "Real Estate", "O": "Real Estate",
    "SBAC": "Real Estate", "AVB": "Real Estate", "EQR": "Real Estate", "INVH": "Real Estate",
    "MAA": "Real Estate", "KIM": "Real Estate", "REXR": "Real Estate", "VTR": "Real Estate",
    "ARE": "Real Estate", "ESS": "Real Estate", "UDR": "Real Estate", "HST": "Real Estate",
    # Utilities
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "SRE": "Utilities",
    "AEP": "Utilities", "D": "Utilities", "EXC": "Utilities", "XEL": "Utilities",
    "ED": "Utilities", "WEC": "Utilities", "ES": "Utilities", "AWK": "Utilities",
    "DTE": "Utilities", "PPL": "Utilities", "PEG": "Utilities", "EIX": "Utilities",
    "FE": "Utilities", "ETR": "Utilities", "AEE": "Utilities", "CNP": "Utilities",
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services", "CMCSA": "Communication Services",
    "VZ": "Communication Services", "T": "Communication Services", "TMUS": "Communication Services",
    "CHTR": "Communication Services", "EA": "Communication Services", "ATVI": "Communication Services",
    "TTWO": "Communication Services", "OMC": "Communication Services", "IPG": "Communication Services",
    "NWSA": "Communication Services", "NWS": "Communication Services", "FOXA": "Communication Services",
    "FOX": "Communication Services", "PARA": "Communication Services",
    # Additional
    "PYPL": "Financials", "SQ": "Financials", "SHOP": "Technology", "UBER": "Consumer Discretionary",
    "LYFT": "Consumer Discretionary", "COIN": "Financials", "HOOD": "Financials",
    "DKNG": "Consumer Discretionary", "RBLX": "Communication Services",
}

# Default to "Unknown" for unmapped tickers
def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker, "Unknown")


# ---------------------------------------------------------------------------
# Scan result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    ticker: str
    sector: str
    sharpe: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    final_capital: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan_universe(
    tickers: List[str] = None,
    start: str = "2020-01-01",
    output_csv: str = "sp500_scan_results.csv",
    **backtest_kwargs,
) -> pd.DataFrame:
    """Run the mean-reversion strategy on each ticker and collect results.

    Parameters
    ----------
    tickers : list of str
        Tickers to scan. Defaults to SP500_TICKERS.
    start : str
        Start date for price data.
    output_csv : str
        Path to save results CSV.
    **backtest_kwargs
        Passed to BacktestConfig.

    Returns
    -------
    pd.DataFrame with one row per ticker, sorted by Sharpe descending.
    """
    if tickers is None:
        tickers = SP500_TICKERS

    results: list[ScanResult] = []
    total = len(tickers)

    logger.info("Starting universe scan: %d tickers, start=%s", total, start)
    logger.info("Backtest params: %s", backtest_kwargs)

    for i, ticker in enumerate(tickers, 1):
        sector = get_sector(ticker)
        logger.info("[%d/%d] Processing %s (%s)...", i, total, ticker, sector)

        try:
            cfg = BacktestConfig(
                tickers=[ticker],
                start=start,
                **backtest_kwargs,
            )
            # Suppress verbose logging during scan
            old_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)

            res = run_backtest(cfg)

            logging.getLogger().setLevel(old_level)

            sr = ScanResult(
                ticker=ticker,
                sector=sector,
                sharpe=res["sharpe"],
                total_return_pct=100 * (res["final_capital"] / cfg.initial_capital - 1),
                max_drawdown_pct=100 * res["max_drawdown"],
                win_rate=100 * res["win_loss"]["win_rate"],
                profit_factor=res["win_loss"]["profit_factor"],
                total_trades=res["win_loss"]["total_trades"],
                final_capital=res["final_capital"],
            )
            results.append(sr)

            logger.info(
                "  ✓ Sharpe=%.2f | Return=%.1f%% | MDD=%.1f%% | Wr=%.0f%% | PF=%.2f | Trades=%d",
                sr.sharpe, sr.total_return_pct, sr.max_drawdown_pct,
                sr.win_rate, sr.profit_factor, sr.total_trades,
            )

        except Exception as e:
            logging.getLogger().setLevel(logging.INFO)
            logger.warning("  ✗ %s failed: %s", ticker, e)
            results.append(ScanResult(
                ticker=ticker, sector=sector, sharpe=0, total_return_pct=0,
                max_drawdown_pct=0, win_rate=0, profit_factor=0, total_trades=0,
                final_capital=0, error=str(e),
            ))

        # Brief pause to be nice to yfinance
        time.sleep(0.5)

    # Build DataFrame
    df = pd.DataFrame([vars(r) for r in results])
    df = df.drop(columns=["error"], errors="ignore")
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info("\nResults saved to %s", output_csv)

    # Print summary
    _print_summary(df)

    return df


def _print_summary(df: pd.DataFrame):
    """Print a human-readable summary of scan results."""
    print("\n" + "=" * 80)
    print("UNIVERSE SCAN SUMMARY")
    print("=" * 80)

    # Top 20 by Sharpe
    print("\n🏆 Top 20 Tickers by Sharpe Ratio:")
    print("-" * 80)
    top20 = df.head(20)[["ticker", "sector", "sharpe", "total_return_pct", "win_rate", "profit_factor", "total_trades"]]
    print(top20.to_string(index=False))

    # Bottom 20 by Sharpe
    print("\n📉 Bottom 20 Tickers by Sharpe Ratio:")
    print("-" * 80)
    bot20 = df.tail(20)[["ticker", "sector", "sharpe", "total_return_pct", "win_rate", "profit_factor", "total_trades"]]
    print(bot20.to_string(index=False))

    # Sector averages
    print("\n📊 Sector Averages (mean across tickers):")
    print("-" * 80)
    sector_stats = df.groupby("sector").agg({
        "sharpe": "mean",
        "total_return_pct": "mean",
        "win_rate": "mean",
        "profit_factor": "mean",
        "total_trades": "mean",
        "ticker": "count",
    }).rename(columns={"ticker": "count"}).sort_values("sharpe", ascending=False)
    print(sector_stats.to_string())

    # Key statistics
    print("\n📈 Portfolio-Level Statistics:")
    print("-" * 80)
    print(f"  Tickers scanned:     {len(df)}")
    print(f"  Mean Sharpe:         {df['sharpe'].mean():.2f}")
    print(f"  Median Sharpe:       {df['sharpe'].median():.2f}")
    print(f"  Std Sharpe:          {df['sharpe'].std():.2f}")
    print(f"  % Positive Sharpe:   {100 * (df['sharpe'] > 0).mean():.1f}%")
    print(f"  Mean Win Rate:       {df['win_rate'].mean():.1f}%")
    print(f"  Mean Profit Factor:  {df['profit_factor'].mean():.2f}")
    print(f"  Mean Total Return:   {df['total_return_pct'].mean():.2f}%")

    # Tickers with most trades
    print("\n🔄 Most Active Tickers (by trade count):")
    print("-" * 80)
    most_active = df.nlargest(20, "total_trades")[["ticker", "sector", "total_trades", "sharpe", "win_rate"]]
    print(most_active.to_string(index=False))

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan S&P 500 universe with mean-reversion strategy")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--output", type=str, default="sp500_scan_results.csv", help="Output CSV path")
    parser.add_argument("--tickers", type=str, nargs="*", default=None, help="Specific tickers to scan (default: all S&P 500)")
    parser.add_argument("--zscore", type=float, default=2.0, help="Z-score threshold")
    parser.add_argument("--rsi-oversold", type=float, default=25.0, help="RSI oversold level")
    parser.add_argument("--rsi-overbought", type=float, default=75.0, help="RSI overbought level")
    parser.add_argument("--position-size", type=float, default=0.10, help="Position size fraction")
    parser.add_argument("--atr-mult", type=float, default=2.0, help="ATR stop multiplier")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    scan_universe(
        tickers=args.tickers,
        start=args.start,
        output_csv=args.output,
        zscore_threshold=args.zscore,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        position_size_pct=args.position_size,
        atr_stop_multiplier=args.atr_mult,
    )
