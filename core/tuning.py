"""
Tuning Module: Parameter optimization for Phase B signals.

Uses a grid-search approach to find optimal signal thresholds that 
maximize a risk-adjusted metric (Composite Score) while penalizing 
excessive drawdown and low trade frequency.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from core.backtest import run_backtest, BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    params: dict
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    profit_factor: float
    score: float  # composite metric


def _composite_score(sharpe: float, mdd: float, num_trades: int, profit_factor: float) -> float:
    """Custom scoring function to find the 'equilibrium point'.

    We want:
    1. High Sharpe (reward risk-adjusted returns)
    2. Low Drawdown (penalize large drops)
    3. Reasonable Trade Count (penalize < 10 trades as statistically insignificant)
    4. Profit Factor > 1.0 (essential for a viable edge)
    """
    # Handle edge cases
    if num_trades < 5:
        return -999.0
    if profit_factor < 0.8:
        return -999.0

    # Normalize components
    sharpe_comp = sharpe
    mdd_comp = max(mdd, -0.25) * 2.0  # penalize drawdown, capped at -25%
    trade_comp = min(num_trades / 40.0, 1.5)  # reward up to 40 trades
    pf_comp = min(profit_factor, 2.5) / 1.25  # reward up to 2.5 PF

    # Weighted sum: Prioritize Sharpe and PF, penalize MDD
    return (0.5 * sharpe_comp) - (0.4 * abs(mdd_comp)) + (0.05 * trade_comp) + (0.3 * pf_comp)


def optimize_signals(
    tickers: List[str] = ["VOO"],
    start: str = "2024-01-01",
) -> TuningResult:
    """Run a grid search over key signal parameters."""
    
    # Parameter Grid - refined based on initial diagnosis
    zscore_thresholds = [1.5, 2.0]
    rsi_oversold = [25, 30]
    rsi_overbought = [75, 80]
    position_sizes = [0.10, 0.15]
    atr_multipliers = [1.5, 2.0, 2.5]

    best_result = None
    best_score = -999

    total_combos = len(zscore_thresholds) * len(rsi_oversold) * len(rsi_overbought) * len(position_sizes) * len(atr_multipliers)
    logger.info(f"🚀 Starting grid search over {total_combos} combinations...")

    combo_count = 0
    for z_thresh, rsi_os, rsi_ob, pos_size, atr_mult in itertools.product(
        zscore_thresholds, rsi_oversold, rsi_overbought, position_sizes, atr_multipliers
    ):
        combo_count += 1
        try:
            cfg = BacktestConfig(
                tickers=tickers,
                start=start,
                position_size_pct=pos_size,
                zscore_threshold=z_thresh,
                rsi_oversold=rsi_os,
                rsi_overbought=rsi_ob,
                atr_stop_multiplier=atr_mult,
            )
            
            # Suppress backtest logging for tuning speed
            logging.getLogger("backtest").setLevel(logging.ERROR)
            logging.getLogger("signal_engine").setLevel(logging.ERROR)
            logging.getLogger("price_collector").setLevel(logging.ERROR)
            
            results = run_backtest(cfg)
            
            score = _composite_score(
                sharpe=results["sharpe"],
                mdd=results["max_drawdown"],
                num_trades=results["win_loss"]["total_trades"],
                profit_factor=results["win_loss"]["profit_factor"]
            )

            if score > best_score:
                best_score = score
                best_result = TuningResult(
                    params={
                        "z_thresh": z_thresh,
                        "rsi_os": rsi_os,
                        "rsi_ob": rsi_ob,
                        "pos_size": pos_size,
                        "atr_mult": atr_mult
                    },
                    sharpe=results["sharpe"],
                    total_return=(results["final_capital"] / cfg.initial_capital - 1) * 100,
                    max_drawdown=results["max_drawdown"] * 100,
                    win_rate=results["win_loss"]["win_rate"] * 100,
                    num_trades=results["win_loss"]["total_trades"],
                    profit_factor=results["win_loss"]["profit_factor"],
                    score=score
                )
                logger.info(f"✨ New best (Combo {combo_count}): Score={score:.2f} | "
                            f"Z={z_thresh}, RSI_OS={rsi_os}, RSI_OB={rsi_ob}, Pos={pos_size}, ATR={atr_mult} | "
                            f"Sharpe={results['sharpe']:.2f}, MDD={results['max_drawdown']:.2%}")

        except Exception as e:
            logger.warning(f"Failed combo: {e}")

    if best_result:
        logger.info("🏆 Optimization Complete. Best Parameters:")
        logger.info(f"   Z-Score Threshold: {best_result.params['z_thresh']}")
        logger.info(f"   RSI Oversold:      {best_result.params['rsi_os']}")
        logger.info(f"   RSI Overbought:    {best_result.params['rsi_ob']}")
        logger.info(f"   Position Size:     {best_result.params['pos_size']}")
        logger.info(f"   ATR Stop Multi:    {best_result.params.get('atr_mult', 'N/A')}")
        logger.info(f"   Composite Score:   {best_result.score:.4f}")
        logger.info(f"   Sharpe:            {best_result.sharpe:.2f}")
        logger.info(f"   Total Return:      {best_result.total_return:.2f}%")
        logger.info(f"   Max Drawdown:      {best_result.max_drawdown:.2f}%")
        logger.info(f"   Win Rate:          {best_result.win_rate:.2f}%")
        logger.info(f"   Profit Factor:     {best_result.profit_factor:.2f}")
        logger.info(f"   Total Trades:      {best_result.num_trades}")
    
    return best_result or TuningResult({}, 0, 0, 0, 0, 0, 0, -999)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    optimize_signals()
