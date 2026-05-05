"""
Vectorized Backtesting Engine for Mean-Reversion Signals.

Simulates a long/short strategy based on signal entries and exits,
computing PnL, Sharpe ratio, max drawdown, and win/loss statistics.

Key design decisions
--------------------
* **Signal-to-trade mapping**:
    - ``LONG_REVERSION`` / ``LONG_BOUNCE``  →  enter long, exit on neutral or opposite
    - ``SHORT_REVERSION`` / ``SHORT_BOUNCE`` →  enter short, exit on neutral or opposite
* **Position sizing**: Equal-dollar per trade (1 unit of capital per signal).
* **Transaction costs**: Configurable flat fee + slippage (bps of notional).
* **No look-ahead**: Signals are computed on data available *at* each date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.price_collector import PriceDataCollector
from core.signal_engine import generate_signals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Backtest hyper-parameters."""
    tickers: list[str] = field(default_factory=lambda: ["VOO"])
    start: str = "2024-01-01"
    end: str | None = None          # None → use collector default
    initial_capital: float = 100_000.0
    commission_per_trade: float = 1.0   # flat $ per trade
    slippage_bps: float = 5.0           # basis points of trade notional
    position_size_pct: float = 0.10     # fraction of capital per open position
    
    # Signal Parameters
    zscore_threshold: float = 2.0
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    
    # Risk Management
    atr_stop_multiplier: float = 2.0    # Stop loss at Entry +/- (ATR * multiplier)


# ---------------------------------------------------------------------------
# Trade log builder
# ---------------------------------------------------------------------------

def _build_trades(signals: pd.DataFrame, atr_stop_multiplier: float = 2.0) -> pd.DataFrame:
    """Convert signal series into entry/exit trade pairs.

    Returns a DataFrame with columns:
        ticker, entry_date, entry_price, entry_signal,
        exit_date, exit_price, exit_reason, pnl_pct, pnl_dollar
    """
    trades: list[dict] = []

    for ticker, grp in signals.groupby("ticker", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        position: str | None = None       # "long" | "short" | None
        entry_idx: int | None = None
        entry_price: float = np.nan

        for idx, row in grp.iterrows():
            sig = row["signal"]
            price = row["close"]

            # --- Open a new position ---
            if position is None:
                if sig in ("LONG_REVERSION", "LONG_BOUNCE"):
                    position = "long"
                    entry_idx = idx
                    entry_price = price
                elif sig in ("SHORT_REVERSION", "SHORT_BOUNCE"):
                    position = "short"
                    entry_idx = idx
                    entry_price = price
                continue

            # --- Close existing position ---
            close_trade = False
            exit_reason = "SIGNAL_REVERSAL"

            # 1. Signal Reversal
            if position == "long":
                if sig in ("SHORT_REVERSION", "SHORT_BOUNCE", "NEUTRAL"):
                    close_trade = True
            elif position == "short":
                if sig in ("LONG_REVERSION", "LONG_BOUNCE", "NEUTRAL"):
                    close_trade = True
            
            # 2. ATR Trailing Stop (Risk Management)
            if not close_trade and "atr_14" in row:
                atr = row["atr_14"]
                stop_dist = atr * atr_stop_multiplier
                if position == "long" and price < (entry_price - stop_dist):
                    close_trade = True
                    exit_reason = "ATR_STOP_LOSS"
                elif position == "short" and price > (entry_price + stop_dist):
                    close_trade = True
                    exit_reason = "ATR_STOP_LOSS"

            if close_trade:
                exit_price = price
                # PnL as percentage return
                if position == "long":
                    pnl_pct = (exit_price / entry_price) - 1.0
                else:
                    pnl_pct = 1.0 - (exit_price / entry_price)

                trades.append({
                    "ticker": ticker,
                    "entry_date": grp.loc[entry_idx, "date"],
                    "entry_price": entry_price,
                    "entry_signal": grp.loc[entry_idx, "signal"],
                    "exit_date": row["date"],
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl_pct": pnl_pct,
                    "pnl_dollar": pnl_pct,  # normalised; dollar PnL computed later
                })
                position = None
                entry_idx = None
                entry_price = np.nan

        # Force-close any open position at end of series
        if position is not None and entry_idx is not None:
            last = grp.iloc[-1]
            exit_price = last["close"]
            if position == "long":
                pnl_pct = (exit_price / entry_price) - 1.0
            else:
                pnl_pct = 1.0 - (exit_price / entry_price)
            trades.append({
                "ticker": ticker,
                "entry_date": grp.loc[entry_idx, "date"],
                "entry_price": entry_price,
                "entry_signal": grp.loc[entry_idx, "signal"],
                "exit_date": last["date"],
                "exit_price": exit_price,
                "exit_reason": "END_OF_DATA",
                "pnl_pct": pnl_pct,
                "pnl_dollar": pnl_pct,
            })

    if not trades:
        return pd.DataFrame(columns=[
            "ticker", "entry_date", "entry_price", "entry_signal",
            "exit_date", "exit_price", "exit_reason", "pnl_pct", "pnl_dollar",
        ])

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------

def _apply_costs(
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Deduct commission and slippage from each trade's PnL."""
    df = trades.copy()
    notional = df["entry_price"]  # per-unit notional
    commission = config.commission_per_trade
    slippage = config.slippage_bps / 10_000.0 * notional
    df["pnl_dollar"] = df["pnl_pct"] * config.initial_capital * config.position_size_pct
    df["cost"] = commission + slippage
    df["pnl_net"] = df["pnl_dollar"] - df["cost"]
    return df


def _portfolio_curve(trades: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    """Build a daily equity curve from realised trade PnL."""
    if trades.empty:
        return pd.Series(dtype=float)

    # Aggregate daily PnL
    all_exits = trades.groupby("exit_date")["pnl_net"].sum()
    dates = pd.date_range(trades["entry_date"].min(), trades["exit_date"].max(), freq="D")
    daily_pnl = all_exits.reindex(dates, fill_value=0.0)
    equity = config.initial_capital + daily_pnl.cumsum()
    return equity


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def _sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio."""
    if daily_returns.std() == 0:
        return 0.0
    return float(np.sqrt(252) * (daily_returns.mean() - risk_free_rate) / daily_returns.std())


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction of peak."""
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def _win_loss_stats(trades: pd.DataFrame) -> dict:
    wins = trades[trades["pnl_net"] > 0]
    losses = trades[trades["pnl_net"] <= 0]
    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades) if len(trades) else 0.0,
        "avg_win": float(wins["pnl_net"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl_net"].mean()) if len(losses) else 0.0,
        "profit_factor": float(wins["pnl_net"].sum() / abs(losses["pnl_net"].sum()))
            if len(losses) and losses["pnl_net"].sum() != 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(config: BacktestConfig | None = None, **kwargs) -> dict:
    """Run the full backtest pipeline and return a results summary.

    Parameters
    ----------
    config : BacktestConfig, optional
    **kwargs : passed to ``BacktestConfig`` if *config* is None.

    Returns
    -------
    dict with keys:
        trades (DataFrame), equity_curve (Series), sharpe (float),
        max_drawdown (float), win_loss (dict), final_capital (float)
    """
    cfg = config or BacktestConfig(**kwargs)

    # 1. Fetch + signal generation
    logger.info("Fetching price data for %s …", cfg.tickers)
    collector = PriceDataCollector(tickers=cfg.tickers, start=cfg.start, end=cfg.end)
    prices = collector.collect()

    logger.info("Generating signals …")
    signals = generate_signals(
        prices,
        zscore_threshold=cfg.zscore_threshold,
        rsi_oversold=cfg.rsi_oversold,
        rsi_overbought=cfg.rsi_overbought,
    )

    # 2. Build trades
    logger.info("Building trade log …")
    trades = _build_trades(signals, cfg.atr_stop_multiplier)
    trades = _apply_costs(trades, cfg)

    # 3. Equity curve
    equity = _portfolio_curve(trades, cfg)
    daily_returns = equity.pct_change().dropna()

    # 4. Metrics
    sharpe = _sharpe_ratio(daily_returns)
    mdd = _max_drawdown(equity)
    wl = _win_loss_stats(trades)
    final_capital = float(equity.iloc[-1]) if not equity.empty else cfg.initial_capital

    results = {
        "config": cfg,
        "trades": trades,
        "equity_curve": equity,
        "daily_returns": daily_returns,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_loss": wl,
        "final_capital": final_capital,
        "signals": signals,  # keep for inspection
    }

    # Log summary
    logger.info("=== Backtest Summary ===")
    logger.info("Final capital:      $%12.2f", final_capital)
    logger.info("Total return:       %8.2f%%", 100 * (final_capital / cfg.initial_capital - 1))
    logger.info("Sharpe ratio:       %8.2f", sharpe)
    logger.info("Max drawdown:       %8.2f%%", 100 * mdd)
    logger.info("Total trades:       %8d", wl["total_trades"])
    logger.info("Win rate:           %8.2f%%", 100 * wl["win_rate"])
    logger.info("Profit factor:      %8.2f", wl["profit_factor"])

    return results


def run_backtest_with_stats(
    config: BacktestConfig | None = None,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
    **kwargs,
) -> dict:
    """Run backtest and append non-parametric statistical inference.

    Parameters
    ----------
    config : BacktestConfig, optional
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.
    n_permutations : int
        Number of permutations for signal-edge test.
    **kwargs : passed to ``BacktestConfig`` if *config* is None.

    Returns
    -------
    dict
        Same as ``run_backtest()`` plus a ``"statistical_report"`` key
        containing the full output of ``statistical_inference.generate_statistical_report()``.
    """
    from statistical_inference import generate_statistical_report

    results = run_backtest(config=config, **kwargs)

    logger.info("\n=== Running Non-Parametric Statistical Inference ===")
    stats_report = generate_statistical_report(
        results,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
    )
    results["statistical_report"] = stats_report

    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest with optional non-parametric stats")
    parser.add_argument("--stats", action="store_true", help="Run bootstrap + permutation tests")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap resamples")
    parser.add_argument("--n-permutations", type=int, default=5000, help="Permutation test iterations")
    parser.add_argument("--ticker", type=str, default="VOO", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.stats:
        results = run_backtest_with_stats(
            tickers=[args.ticker],
            start=args.start,
            initial_capital=100_000.0,
            commission_per_trade=1.0,
            slippage_bps=5.0,
            position_size_pct=0.10,
            n_bootstrap=args.n_bootstrap,
            n_permutations=args.n_permutations,
        )
    else:
        results = run_backtest(
            tickers=[args.ticker],
            start=args.start,
            initial_capital=100_000.0,
            commission_per_trade=1.0,
            slippage_bps=5.0,
            position_size_pct=0.10,
        )

    print("\n--- Last 10 Trades ---")
    print(results["trades"][["ticker", "entry_date", "exit_date", "pnl_net"]].tail(10).to_string(index=False))

    if args.stats and "statistical_report" in results:
        print("\n=== Statistical Report Summary ===")
        summary = results["statistical_report"]["summary"]
        print(summary["conclusion"])
