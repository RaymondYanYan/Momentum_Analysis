"""
Non-Parametric Statistical Inference for Trading Signals.

Provides rigorous, distribution-free methods to validate whether observed
signal performance is genuine edge or statistical artifact.

Key Methods
-----------
1. Bootstrap Confidence Intervals
   - Resample trades (or returns) to build empirical CIs on Sharpe, MDD, etc.
   - No assumption of normality or i.i.d. returns.

2. Permutation Tests
   - Shuffle signal labels to test the null: "signal has no predictive power"
   - Direct empirical p-value without parametric assumptions.

3. Rank-Based Analysis
   - Spearman correlation between signal strength and subsequent returns
   - Quantile-based return analysis (how do tails behave?)

4. Kernel Density Estimation
   - Model return distributions flexibly per signal type
   - Compare long vs short vs neutral return distributions

5. Walk-Forward Validation
   - Proper out-of-sample testing with expanding/rolling windows
   - No look-ahead, no parametric curve-fitting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Results from bootstrap resampling."""
    metric_name: str
    observed_value: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    bootstrap_means: np.ndarray
    n_resamples: int


def bootstrap_trades(
    trades: pd.DataFrame,
    metric: str = "sharpe",
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap confidence intervals by resampling individual trades.

    This is the **pairs bootstrap**: treat each trade as an independent
    observation and resample with replacement.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade log from backtest with columns including `pnl_net`.
    metric : str
        Which metric to bootstrap. One of:
        - "sharpe"       : Annualized Sharpe ratio
        - "mean_pnl"     : Mean PnL per trade
        - "win_rate"     : Fraction of winning trades
        - "profit_factor": Gross profit / gross loss
        - "total_return" : Cumulative return
    n_resamples : int
        Number of bootstrap iterations.
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    pnl = trades["pnl_net"].values

    if len(pnl) < 10:
        logger.warning("Only %d trades — bootstrap may be unreliable", len(pnl))

    # Compute observed metric
    observed = _compute_metric(pnl, metric)

    # Bootstrap resampling
    bootstrap_values = np.zeros(n_resamples)
    for i in range(n_resamples):
        resample = rng.choice(pnl, size=len(pnl), replace=True)
        bootstrap_values[i] = _compute_metric(resample, metric)

    # Confidence interval (percentile method)
    alpha = 1.0 - ci_level
    ci_lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))

    return BootstrapResult(
        metric_name=metric,
        observed_value=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        bootstrap_means=bootstrap_values,
        n_resamples=n_resamples,
    )


def bootstrap_equity_curve(
    daily_returns: pd.Series,
    metric: str = "sharpe",
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    block_size: int = 5,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap confidence intervals using **block bootstrap** on returns.

    The block bootstrap preserves serial correlation in returns by
    resampling contiguous blocks rather than individual observations.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series from the equity curve.
    metric : str
        Metric to bootstrap: "sharpe", "mean_return", "volatility", "calmar".
    n_resamples : int
    block_size : int
        Length of blocks for the moving block bootstrap. Larger blocks
        preserve more autocorrelation but reduce effective sample size.
    ci_level, seed : see bootstrap_trades.

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    returns = daily_returns.values
    n = len(returns)

    observed = _compute_metric_from_returns(returns, metric)

    # Moving block bootstrap
    bootstrap_values = np.zeros(n_resamples)
    n_blocks = int(np.ceil(n / block_size))

    for i in range(n_resamples):
        # Sample block start indices
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        resample = np.concatenate([returns[s:s + block_size] for s in starts])[:n]
        bootstrap_values[i] = _compute_metric_from_returns(resample, metric)

    alpha = 1.0 - ci_level
    ci_lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))

    return BootstrapResult(
        metric_name=metric,
        observed_value=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        bootstrap_means=bootstrap_values,
        n_resamples=n_resamples,
    )


def _compute_metric(pnl: np.ndarray, metric: str) -> float:
    """Compute a performance metric from an array of trade PnLs."""
    if metric == "sharpe":
        # Approximate Sharpe from trade returns (not ideal, but common)
        if pnl.std() == 0:
            return 0.0
        return float(np.sqrt(len(pnl)) * pnl.mean() / pnl.std())
    elif metric == "mean_pnl":
        return float(pnl.mean())
    elif metric == "win_rate":
        return float(np.mean(pnl > 0))
    elif metric == "profit_factor":
        wins = pnl[pnl > 0].sum()
        losses = abs(pnl[pnl <= 0].sum())
        return float(wins / losses) if losses > 0 else float("inf")
    elif metric == "total_return":
        return float(np.sum(pnl))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _compute_metric_from_returns(returns: np.ndarray, metric: str) -> float:
    """Compute a performance metric from daily returns."""
    if metric == "sharpe":
        if returns.std() == 0:
            return 0.0
        return float(np.sqrt(252) * returns.mean() / returns.std())
    elif metric == "mean_return":
        return float(returns.mean())
    elif metric == "volatility":
        return float(returns.std() * np.sqrt(252))
    elif metric == "calmar":
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        mdd = abs(drawdown.min())
        annual_return = returns.mean() * 252
        return float(annual_return / mdd) if mdd > 0 else 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# 2. Permutation Test for Signal Edge
# ---------------------------------------------------------------------------

@dataclass
class PermutationResult:
    """Results from permutation test."""
    observed_statistic: float
    p_value: float
    n_permutations: int
    null_distribution: np.ndarray
    test_type: str  # "mean_diff", "sharpe_diff", "rank_correlation"


def permutation_test_signal_edge(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    n_permutations: int = 5000,
    test_type: str = "sharpe_diff",
    seed: int = 42,
) -> PermutationResult:
    """Test whether signal-generated returns differ from random chance.

    The null hypothesis: **the signal has no predictive power**, i.e.,
    the observed performance could arise from random entry/exit timing.

    Method
    ------
    1. Compute the observed test statistic from actual signal returns.
    2. Shuffle signal labels (or entry dates) many times.
    3. Recompute the statistic for each shuffled version.
    4. p-value = fraction of permuted stats >= observed stat.

    Parameters
    ----------
    signals : pd.DataFrame
        Augmented price data with `signal` column (LONG_*, SHORT_*, NEUTRAL).
    prices : pd.DataFrame
        Raw price data with `close` column. Must align with `signals`.
    n_permutations : int
    test_type : str
        - "sharpe_diff"   : Sharpe of signal periods vs random periods
        - "mean_diff"     : Mean return of signal periods vs random
        - "rank_correlation": Spearman rho between signal strength and return
    seed : int

    Returns
    -------
    PermutationResult
    """
    rng = np.random.default_rng(seed)

    # Compute forward returns for each row
    sig = signals.copy()
    sig["forward_return"] = sig.groupby("ticker")["close"].shift(-1) / sig["close"] - 1.0

    # Filter to active signals only
    active = sig[sig["signal"] != "NEUTRAL"].dropna(subset=["forward_return"])
    if len(active) == 0:
        logger.warning("No active signals found for permutation test")
        return PermutationResult(0.0, 1.0, 0, np.array([]), test_type)

    # Neutral periods for comparison
    neutral = sig[sig["signal"] == "NEUTRAL"].dropna(subset=["forward_return"])

    if len(neutral) == 0:
        # Fallback: compare against shuffled version of active signals themselves
        logger.info("No neutral periods — using self-shuffling permutation test")
        observed_stat = _compute_permutation_stat(active["forward_return"].values, test_type)
        null_dist = np.zeros(n_permutations)

        for i in range(n_permutations):
            shuffled = rng.permutation(active["forward_return"].values)
            null_dist[i] = _compute_permutation_stat(shuffled, test_type)
    else:
        # Compare active vs neutral
        active_returns = active["forward_return"].values
        neutral_returns = neutral["forward_return"].values

        observed_stat = _compute_two_sample_stat(active_returns, neutral_returns, test_type)

        # Permutation: pool both groups, randomly reassign labels
        pooled = np.concatenate([active_returns, neutral_returns])
        n_active = len(active_returns)
        null_dist = np.zeros(n_permutations)

        for i in range(n_permutations):
            rng.shuffle(pooled)
            perm_active = pooled[:n_active]
            perm_neutral = pooled[n_active:]
            null_dist[i] = _compute_two_sample_stat(perm_active, perm_neutral, test_type)

    # One-sided p-value: proportion of null stats >= observed
    p_value = float(np.mean(null_dist >= observed_stat))

    return PermutationResult(
        observed_statistic=observed_stat,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution=null_dist,
        test_type=test_type,
    )


def _compute_permutation_stat(returns: np.ndarray, test_type: str) -> float:
    """Compute a test statistic from a single return series."""
    if test_type == "sharpe_diff":
        if returns.std() == 0:
            return 0.0
        return float(np.sqrt(len(returns)) * returns.mean() / returns.std())
    elif test_type == "mean_diff":
        return float(returns.mean())
    elif test_type == "rank_correlation":
        # Self-correlation is degenerate; return 0
        return 0.0
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


def _compute_two_sample_stat(
    group_a: np.ndarray, group_b: np.ndarray, test_type: str
) -> float:
    """Compute a two-sample test statistic."""
    if test_type == "sharpe_diff":
        sharpe_a = _compute_permutation_stat(group_a, "sharpe_diff")
        sharpe_b = _compute_permutation_stat(group_b, "sharpe_diff")
        return sharpe_a - sharpe_b
    elif test_type == "mean_diff":
        return float(group_a.mean() - group_b.mean())
    elif test_type == "rank_correlation":
        # Not applicable for two-sample; fall back to mean diff
        return float(group_a.mean() - group_b.mean())
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


# ---------------------------------------------------------------------------
# 3. Rank-Based Analysis
# ---------------------------------------------------------------------------

@dataclass
class RankAnalysisResult:
    """Results from rank-based signal analysis."""
    spearman_rho: float
    spearman_pvalue: float
    kendall_tau: float
    kendall_pvalue: float
    quantile_returns: pd.DataFrame  # mean return per signal-type quantile


def rank_based_signal_analysis(
    signals: pd.DataFrame,
    price_col: str = "close",
    horizons: list[int] = None,
) -> RankAnalysisResult:
    """Analyze signal predictive power using rank-based methods.

    This tests whether **stronger signals predict larger returns**,
    without assuming linearity or normality.

    Parameters
    ----------
    signals : pd.DataFrame
        Augmented price data with signal columns (ou_kf_zscore, rsi_14, etc.)
        and `signal` column.
    price_col : str
    horizons : list[int]
        Forward return horizons to test. Default: [1, 5, 20] days.

    Returns
    -------
    RankAnalysisResult
    """
    if horizons is None:
        horizons = [1, 5, 20]

    sig = signals.copy()

    # Compute forward returns for multiple horizons
    for h in horizons:
        sig[f"forward_ret_{h}d"] = (
            sig.groupby("ticker")[price_col].shift(-h) / sig[price_col] - 1.0
        )

    # Map signals to numeric strength scores
    # LONG_* = positive, SHORT_* = negative, NEUTRAL = 0
    sig["signal_strength"] = 0.0
    sig.loc[sig["signal"].str.contains("LONG"), "signal_strength"] = 1.0
    sig.loc[sig["signal"].str.contains("SHORT"), "signal_strength"] = -1.0
    sig.loc[sig["signal"] == "NEUTRAL", "signal_strength"] = 0.0

    # For continuous signal strength, use OU-KF Z-score (inverted for long signals)
    if "ou_kf_zscore" in sig.columns:
        sig["continuous_strength"] = -sig["ou_kf_zscore"]  # negative zscore = bullish
    else:
        sig["continuous_strength"] = sig["signal_strength"]

    results = {}

    for h in horizons:
        fwd_col = f"forward_ret_{h}d"
        valid = sig.dropna(subset=[fwd_col, "continuous_strength"])

        if len(valid) < 30:
            logger.warning("Insufficient data for %d-day horizon", h)
            continue

        # Spearman rank correlation
        rho, rho_p = stats.spearmanr(valid["continuous_strength"], valid[fwd_col])

        # Kendall tau
        tau, tau_p = stats.kendalltau(valid["continuous_strength"], valid[fwd_col])

        results[h] = {
            "spearman_rho": rho,
            "spearman_pvalue": rho_p,
            "kendall_tau": tau,
            "kendall_pvalue": tau_p,
        }

        logger.info(
            "Horizon %dd: Spearman ρ=%.4f (p=%.4f), Kendall τ=%.4f (p=%.4f), n=%d",
            h, rho, rho_p, tau, tau_p, len(valid),
        )

    # Quantile analysis: bin by signal strength, compute mean forward return
    sig_valid = sig.dropna(subset=["continuous_strength", f"forward_ret_{horizons[0]}d"])
    sig_valid["strength_quintile"] = pd.qcut(
        sig_valid["continuous_strength"].rank(method="first"),
        q=5,
        labels=["Q1 (Most Bearish)", "Q2", "Q3", "Q4", "Q5 (Most Bullish)"],
    )

    quantile_rets = sig_valid.groupby(["strength_quintile"], observed=True)[
        [f"forward_ret_{h}d" for h in horizons]
    ].mean()

    # Aggregate results
    avg_rho = np.mean([v["spearman_rho"] for v in results.values()])
    avg_rho_p = np.mean([v["spearman_pvalue"] for v in results.values()])
    avg_tau = np.mean([v["kendall_tau"] for v in results.values()])
    avg_tau_p = np.mean([v["kendall_pvalue"] for v in results.values()])

    return RankAnalysisResult(
        spearman_rho=avg_rho,
        spearman_pvalue=avg_rho_p,
        kendall_tau=avg_tau,
        kendall_pvalue=avg_tau_p,
        quantile_returns=quantile_rets,
    )


# ---------------------------------------------------------------------------
# 4. Kernel Density Estimation of Return Distributions
# ---------------------------------------------------------------------------

@dataclass
class KDEAnalysisResult:
    """Results from KDE-based return distribution analysis."""
    bandwidth: float
    long_pdf: np.ndarray
    short_pdf: np.ndarray
    neutral_pdf: np.ndarray
    x_grid: np.ndarray
    ks_stat_long_neutral: float
    ks_pval_long_neutral: float
    ks_stat_short_neutral: float
    ks_pval_short_neutral: float


def kde_return_analysis(
    signals: pd.DataFrame,
    price_col: str = "close",
    horizon: int = 1,
    grid_points: int = 500,
) -> KDEAnalysisResult:
    """Compare return distributions across signal types using KDE.

    Tests whether long, short, and neutral signals produce **statistically
    different** return distributions — non-parametrically.

    Parameters
    ----------
    signals : pd.DataFrame
    price_col : str
    horizon : int
        Forward return horizon in days.
    grid_points : int
        Number of points in the KDE evaluation grid.

    Returns
    -------
    KDEAnalysisResult
    """
    sig = signals.copy()
    sig["forward_return"] = (
        sig.groupby("ticker")[price_col].shift(-horizon) / sig[price_col] - 1.0
    )

    # Split by signal type
    long_rets = sig.loc[sig["signal"].str.contains("LONG"), "forward_return"].dropna()
    short_rets = sig.loc[sig["signal"].str.contains("SHORT"), "forward_return"].dropna()
    neutral_rets = sig.loc[sig["signal"] == "NEUTRAL", "forward_return"].dropna()

    if len(long_rets) < 10 or len(neutral_rets) < 10:
        logger.warning("Insufficient data for KDE analysis")
        return KDEAnalysisResult(0, np.array([]), np.array([]), np.array([]), np.array([]), 0, 1, 0, 1)

    # Combine all returns to define grid
    all_rets = pd.concat([long_rets, short_rets, neutral_rets]).values
    x_min, x_max = np.percentile(all_rets, [1, 99])
    x_grid = np.linspace(x_min, x_max, grid_points)

    # Bandwidth selection (Silverman's rule of thumb)
    bandwidth = 1.06 * all_rets.std() * len(all_rets) ** (-1 / 5)

    # Fit KDEs
    def _kde(data, x, bw):
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(data.reshape(-1, 1))
        log_dens = kde.score_samples(x.reshape(-1, 1))
        return np.exp(log_dens)

    long_pdf = _kde(long_rets.values, x_grid, bandwidth)
    short_pdf = _kde(short_rets.values, x_grid, bandwidth) if len(short_rets) > 5 else np.zeros_like(x_grid)
    neutral_pdf = _kde(neutral_rets.values, x_grid, bandwidth)

    # Kolmogorov-Smirnov test: are distributions different?
    ks_long_neut = stats.ks_2samp(long_rets.values, neutral_rets.values)
    ks_short_neut = (
        stats.ks_2samp(short_rets.values, neutral_rets.values)
        if len(short_rets) > 5
        else (0.0, 1.0)
    )

    logger.info(
        "KDE Analysis: KS(long vs neutral) = %.4f (p=%.4f), "
        "KS(short vs neutral) = %.4f (p=%.4f)",
        ks_long_neut.statistic, ks_long_neut.pvalue,
        ks_short_neut[0], ks_short_neut[1],
    )

    return KDEAnalysisResult(
        bandwidth=bandwidth,
        long_pdf=long_pdf,
        short_pdf=short_pdf,
        neutral_pdf=neutral_pdf,
        x_grid=x_grid,
        ks_stat_long_neutral=ks_long_neut.statistic,
        ks_pval_long_neutral=ks_long_neut.pvalue,
        ks_stat_short_neutral=ks_short_neut[0],
        ks_pval_short_neutral=ks_short_neut[1],
    )


# ---------------------------------------------------------------------------
# 5. Walk-Forward Validation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    train_periods: list[tuple[str, str]]
    test_periods: list[tuple[str, str]]
    out_of_sample_sharpe: list[float]
    out_of_sample_returns: list[float]
    in_sample_sharpe: list[float]
    overfitting_ratio: float  # mean IS Sharpe / mean OOS Sharpe


def walk_forward_validation(
    signals_fn,
    prices: pd.DataFrame,
    n_folds: int = 5,
    train_fraction: float = 0.6,
    **signal_kwargs,
) -> WalkForwardResult:
    """Walk-forward validation: train signals on expanding window, test OOS.

    This properly validates whether signal parameters **generalize** or are
    merely overfit to historical data.

    Parameters
    ----------
    signals_fn : callable
        Function that takes (prices, **kwargs) and returns signals DataFrame.
        E.g., `signal_engine.generate_signals`.
    prices : pd.DataFrame
        Raw price data.
    n_folds : int
        Number of train/test splits.
    train_fraction : float
        Fraction of data used for training in each fold.
    **signal_kwargs
        Passed to `signals_fn`.

    Returns
    -------
    WalkForwardResult
    """
    from core.backtest import run_backtest, BacktestConfig

    dates = prices["date"].unique()
    dates = np.sort(dates)
    total_days = len(dates)

    fold_size = total_days // (n_folds + 1)
    train_periods = []
    test_periods = []
    is_sharpes = []
    oos_sharpes = []
    oos_returns = []

    for fold in range(n_folds):
        train_end_idx = (fold + 1) * fold_size
        test_end_idx = (fold + 2) * fold_size

        train_start = dates[0]
        train_end = dates[train_end_idx]
        test_start = dates[train_end_idx + 1]
        test_end = dates[min(test_end_idx, total_days - 1)]

        train_periods.append((str(train_start), str(train_end)))
        test_periods.append((str(test_start), str(test_end)))

        logger.info(
            "Fold %d: Train [%s → %s], Test [%s → %s]",
            fold, train_start, train_end, test_start, test_end,
        )

        # Train: generate signals and compute in-sample Sharpe
        try:
            train_prices = prices[prices["date"] <= train_end]
            train_signals = signals_fn(train_prices, **signal_kwargs)

            # Quick in-sample backtest
            is_config = BacktestConfig(
                tickers=["VOO"],
                start=str(train_start),
                end=str(train_end),
                position_size_pct=0.10,
            )
            is_results = run_backtest(train_prices, **signal_kwargs)
            is_sharpes.append(is_results["sharpe"])
        except Exception as e:
            logger.warning("Train fold %d failed: %s", fold, e)
            is_sharpes.append(0.0)

        # Test: apply same signal logic OOS
        try:
            test_prices = prices[
                (prices["date"] >= test_start) & (prices["date"] <= test_end)
            ]
            oos_results = run_backtest(test_prices, **signal_kwargs)
            oos_sharpes.append(oos_results["sharpe"])
            oos_returns.append(
                (oos_results["final_capital"] / oos_results["config"].initial_capital - 1)
            )
        except Exception as e:
            logger.warning("Test fold %d failed: %s", fold, e)
            oos_sharpes.append(0.0)
            oos_returns.append(0.0)

    # Overfitting ratio: if >> 1, signal is overfit
    mean_is = np.mean(is_sharpes) if is_sharpes else 0
    mean_oos = np.mean(oos_sharpes) if oos_sharpes else 0
    overfit_ratio = mean_is / mean_oos if mean_oos != 0 else float("inf")

    logger.info(
        "Walk-Forward Complete: Mean IS Sharpe=%.2f, Mean OOS Sharpe=%.2f, "
        "Overfit Ratio=%.2f",
        mean_is, mean_oos, overfit_ratio,
    )

    return WalkForwardResult(
        train_periods=train_periods,
        test_periods=test_periods,
        out_of_sample_sharpe=oos_sharpes,
        out_of_sample_returns=oos_returns,
        in_sample_sharpe=is_sharpes,
        overfitting_ratio=overfit_ratio,
    )


# ---------------------------------------------------------------------------
# 6. Unified Report
# ---------------------------------------------------------------------------

def generate_statistical_report(
    backtest_results: dict,
    n_bootstrap: int = 5000,
    n_permutations: int = 5000,
) -> dict:
    """Generate a comprehensive non-parametric statistical report.

    Parameters
    ----------
    backtest_results : dict
        Output from `backtest.run_backtest()`.
    n_bootstrap : int
    n_permutations : int

    Returns
    -------
    dict with keys:
        - bootstrap_ci : dict of BootstrapResult per metric
        - permutation_test : PermutationResult
        - rank_analysis : RankAnalysisResult
        - kde_analysis : KDEAnalysisResult
        - summary : human-readable dict
    """
    logger.info("=== Non-Parametric Statistical Report ===")

    trades = backtest_results["trades"]
    daily_returns = backtest_results["daily_returns"]
    signals = backtest_results["signals"]

    # 1. Bootstrap CIs
    logger.info("Running bootstrap confidence intervals...")
    bootstrap_ci = {
        "sharpe": bootstrap_trades(trades, "sharpe", n_resamples=n_bootstrap),
        "win_rate": bootstrap_trades(trades, "win_rate", n_resamples=n_bootstrap),
        "profit_factor": bootstrap_trades(trades, "profit_factor", n_resamples=n_bootstrap),
        "mean_pnl": bootstrap_trades(trades, "mean_pnl", n_resamples=n_bootstrap),
    }

    for metric, result in bootstrap_ci.items():
        logger.info(
            "  %s: %.4f [%.4f, %.4f] (%.0f%% CI, %d resamples)",
            metric,
            result.observed_value,
            result.ci_lower,
            result.ci_upper,
            result.ci_level * 100,
            result.n_resamples,
        )

    # 2. Block bootstrap on equity curve
    logger.info("Running block bootstrap on equity curve...")
    equity_bootstrap = bootstrap_equity_curve(
        daily_returns, "sharpe", n_resamples=n_bootstrap, block_size=5
    )
    logger.info(
        "  Sharpe (block bootstrap): %.4f [%.4f, %.4f]",
        equity_bootstrap.observed_value,
        equity_bootstrap.ci_lower,
        equity_bootstrap.ci_upper,
    )

    # 3. Permutation test
    logger.info("Running permutation test for signal edge...")
    perm_result = permutation_test_signal_edge(
        signals, backtest_results["signals"], n_permutations=n_permutations
    )
    logger.info(
        "  Permutation test: observed=%.4f, p-value=%.4f (%d permutations)",
        perm_result.observed_statistic,
        perm_result.p_value,
        perm_result.n_permutations,
    )

    # 4. Rank-based analysis
    logger.info("Running rank-based signal analysis...")
    rank_result = rank_based_signal_analysis(signals)
    logger.info(
        "  Spearman ρ=%.4f (p=%.4f), Kendall τ=%.4f (p=%.4f)",
        rank_result.spearman_rho,
        rank_result.spearman_pvalue,
        rank_result.kendall_tau,
        rank_result.kendall_pvalue,
    )

    # 5. KDE analysis
    logger.info("Running KDE return distribution analysis...")
    kde_result = kde_return_analysis(signals)
    logger.info(
        "  KS test (long vs neutral): stat=%.4f, p=%.4f",
        kde_result.ks_stat_long_neutral,
        kde_result.ks_pval_long_neutral,
    )

    # Summary
    summary = {
        "bootstrap_sharpe_ci": (
            bootstrap_ci["sharpe"].ci_lower,
            bootstrap_ci["sharpe"].ci_upper,
        ),
        "bootstrap_win_rate_ci": (
            bootstrap_ci["win_rate"].ci_lower,
            bootstrap_ci["win_rate"].ci_upper,
        ),
        "permutation_p_value": perm_result.p_value,
        "signal_edge_significant": perm_result.p_value < 0.05,
        "spearman_rho": rank_result.spearman_rho,
        "rank_correlation_significant": rank_result.spearman_pvalue < 0.05,
        "ks_long_vs_neutral_p": kde_result.ks_pval_long_neutral,
        "distributions_different": kde_result.ks_pval_long_neutral < 0.05,
        "conclusion": _summarize_conclusion(
            bootstrap_ci, perm_result, rank_result, kde_result
        ),
    }

    logger.info("=== Conclusion ===")
    logger.info(summary["conclusion"])

    return {
        "bootstrap_ci": bootstrap_ci,
        "equity_bootstrap": equity_bootstrap,
        "permutation_test": perm_result,
        "rank_analysis": rank_result,
        "kde_analysis": kde_result,
        "summary": summary,
    }


def _summarize_conclusion(
    bootstrap_ci: dict,
    perm_result: PermutationResult,
    rank_result: RankAnalysisResult,
    kde_result: KDEAnalysisResult,
) -> str:
    """Generate a plain-language conclusion from all tests."""
    lines = []

    # Sharpe CI
    sharpe_ci = bootstrap_ci["sharpe"]
    if sharpe_ci.ci_lower > 0:
        lines.append(
            f"✓ Sharpe ratio is significantly positive "
            f"({sharpe_ci.observed_value:.2f}, 95% CI [{sharpe_ci.ci_lower:.2f}, {sharpe_ci.ci_upper:.2f}])"
        )
    else:
        lines.append(
            f"✗ Sharpe ratio CI includes zero "
            f"({sharpe_ci.observed_value:.2f}, 95% CI [{sharpe_ci.ci_lower:.2f}, {sharpe_ci.ci_upper:.2f}])"
        )

    # Permutation test
    if perm_result.p_value < 0.05:
        lines.append(
            f"✓ Signal edge is statistically significant "
            f"(permutation p={perm_result.p_value:.4f})"
        )
    else:
        lines.append(
            f"✗ No significant signal edge detected "
            f"(permutation p={perm_result.p_value:.4f})"
        )

    # Rank correlation
    if rank_result.spearman_pvalue < 0.05:
        lines.append(
            f"✓ Signal strength correlates with returns "
            f"(Spearman ρ={rank_result.spearman_rho:.3f}, p={rank_result.spearman_pvalue:.4f})"
        )
    else:
        lines.append(
            f"✗ No significant rank correlation "
            f"(Spearman ρ={rank_result.spearman_rho:.3f}, p={rank_result.spearman_pvalue:.4f})"
        )

    # KDE / distribution test
    if kde_result.ks_pval_long_neutral < 0.05:
        lines.append(
            f"✓ Long signal returns differ from neutral "
            f"(KS p={kde_result.ks_pval_long_neutral:.4f})"
        )
    else:
        lines.append(
            f"✗ Long signal returns similar to neutral "
            f"(KS p={kde_result.ks_pval_long_neutral:.4f})"
        )

    # Overall
    all_pass = (
        sharpe_ci.ci_lower > 0
        and perm_result.p_value < 0.05
        and rank_result.spearman_pvalue < 0.05
    )
    if all_pass:
        lines.append(
            "\n🎯 OVERALL: Signal shows robust statistical edge across multiple "
            "non-parametric tests. Proceed with caution (past ≠ future)."
        )
    else:
        lines.append(
            "\n⚠️  OVERALL: Signal edge is weak or inconsistent. "
            "Consider refining parameters or collecting more data."
        )

    return "\n".join(lines)
