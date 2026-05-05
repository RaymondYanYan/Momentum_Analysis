"""
Mean Reversion Engine: Rolling Z-Score + Ornstein-Uhlenbeck / Kalman Filter.

Provides two complementary signals:
1. Rolling Z-Score: Simple, interpretable, window-based.
2. OU-KF Z-Score: Adaptive, recursive Kalman Filter on an OU process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize


# ---------------------------------------------------------------------------
# 1. Rolling Z-Score (vectorized)
# ---------------------------------------------------------------------------

def rolling_zscore(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling Z-scores for multiple lookback windows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``price_col`` and be sorted by ``date`` within each
        ``ticker`` group.
    price_col : str
        Column name for the price series.
    windows : list[int]
        Rolling window lengths.  Defaults to ``[20, 50, 200]``.

    Returns
    -------
    pd.DataFrame
        Original *df* with added columns ``zscore_{w}`` for each window *w*.
    """
    if windows is None:
        windows = [20, 50, 200]

    out = df.copy()
    for w in windows:
        grp = out.groupby("ticker", sort=False)[price_col]
        rolling_mean = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
        rolling_std = grp.transform(lambda s: s.rolling(w, min_periods=1).std())
        out[f"zscore_{w}"] = (out[price_col] - rolling_mean) / rolling_std
    return out


# ---------------------------------------------------------------------------
# 2. OU process parameter estimation (MLE via regression)
# ---------------------------------------------------------------------------

def _fit_ou_params(returns: np.ndarray) -> tuple[float, float, float]:
    """Estimate OU parameters (mu, theta, sigma) from a return series.

    Uses the discrete-time OU formulation:
        X_t = mu + phi * (X_{t-1} - mu) + eps
    where phi = exp(-theta * dt).  We estimate via OLS on lagged values.

    Returns (mu, theta, sigma) where:
        mu    – long-term mean
        theta – mean-reversion speed
        sigma – instantaneous volatility
    """
    x = returns.values  # type: ignore[attr-defined]
    x_prev = x[:-1]
    x_curr = x[1:]

    # OLS: x_curr = a + b * x_prev + eps
    # Then mu = a / (1 - b),  phi = b,  theta = -ln(phi)
    n = len(x_prev)
    sum_x = x_prev.sum()
    sum_y = x_curr.sum()
    sum_xy = (x_prev * x_curr).sum()
    sum_xx = (x_prev ** 2).sum()

    denom = n * sum_xx - sum_x ** 2
    if abs(denom) < 1e-15:
        # Degenerate case — revert to naive stats
        return float(np.mean(x)), 1.0, float(np.std(x))

    b = (n * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - b * sum_x) / n

    # Clamp phi to (0, 1) for stationarity
    b = np.clip(b, 0.01, 0.99)

    mu = a / (1 - b)
    theta = -np.log(b)
    sigma = np.std(x_curr - (a + b * x_prev))

    return float(mu), float(theta), float(sigma)


# ---------------------------------------------------------------------------
# 3. 1-D Kalman Filter on OU process (sequential, per ticker)
# ---------------------------------------------------------------------------

def _kalman_filter_ou(
    observations: np.ndarray,
    mu_init: float,
    theta: float,
    sigma: float,
    obs_noise: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a 1-D Kalman Filter tracking the hidden mean of an OU process.

    Parameters
    ----------
    observations : np.ndarray
        Price or spread series.
    mu_init : float
        Initial estimate of the long-term mean.
    theta : float
        Mean-reversion speed.
    sigma : float
        Process noise (innovation std).
    obs_noise : float, optional
        Observation noise std.  If *None*, inferred from data.

    Returns
    -------
    kf_mean : np.ndarray
        Filtered estimate of the hidden mean at each step.
    kf_var : np.ndarray
        Filtered variance (uncertainty) at each step.
    """
    n = len(observations)
    kf_mean = np.zeros(n)
    kf_var = np.zeros(n)

    # Initial state
    x_hat = mu_init
    p = 1.0  # initial variance

    if obs_noise is None:
        obs_noise = float(np.std(observations) * 0.1)

    q = sigma ** 2  # process noise variance
    r = obs_noise ** 2  # observation noise variance

    for t in range(n):
        # --- Prediction ---
        # State transition: mean-reverts toward mu_init
        x_pred = mu_init + np.exp(-theta) * (x_hat - mu_init)
        p_pred = p + q

        # --- Update ---
        z = observations[t]
        y = z - x_pred          # innovation
        s = p_pred + r          # innovation covariance
        k = p_pred / s          # Kalman gain

        x_hat = x_pred + k * y
        p = (1 - k) * p_pred

        kf_mean[t] = x_hat
        kf_var[t] = p

    return kf_mean, kf_var


# ---------------------------------------------------------------------------
# 4. Public API: OU-KF Z-Score
# ---------------------------------------------------------------------------

def ou_kf_zscore(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute adaptive Z-scores via OU-process Kalman Filter.

    For each ticker:
    1. Estimate OU parameters (mu, theta, sigma) from historical data.
    2. Run a 1-D Kalman Filter to track the hidden mean.
    3. Z-score = (price - kf_mean) / sqrt(kf_var).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ticker``, ``date``, and ``price_col``.  Sorted by
        ``date`` within each ticker.
    price_col : str
        Column name for the price series.

    Returns
    -------
    pd.DataFrame
        Original *df* with added columns:
        - ``ou_kf_mean``  : Kalman-filtered estimate of the equilibrium price
        - ``ou_kf_std``   : sqrt of filtered variance (uncertainty)
        - ``ou_kf_zscore``: Normalised residual = (price - mean) / std
    """
    results: list[pd.DataFrame] = []

    for ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.copy()
        prices = grp[price_col].values.astype(np.float64)

        if len(prices) < 30:
            # Not enough data — fill with NaN
            grp["ou_kf_mean"] = np.nan
            grp["ou_kf_std"] = np.nan
            grp["ou_kf_zscore"] = np.nan
            results.append(grp)
            continue

        # Fit OU params on log-prices for better stationarity
        log_prices = np.log(prices)
        mu, theta, sigma = _fit_ou_params(pd.Series(log_prices))

        # Run KF on log-prices
        kf_mean_log, kf_var_log = _kalman_filter_ou(
            log_prices, mu_init=mu, theta=theta, sigma=sigma
        )

        # Convert back to price space
        kf_mean = np.exp(kf_mean_log)
        kf_std = np.exp(kf_mean_log) * np.sqrt(kf_var_log)  # delta method

        grp["ou_kf_mean"] = kf_mean
        grp["ou_kf_std"] = kf_std
        grp["ou_kf_zscore"] = (prices - kf_mean) / np.maximum(kf_std, 1e-10)

        results.append(grp)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Combined mean-reversion signal
# ---------------------------------------------------------------------------

def mean_reversion_signals(
    df: pd.DataFrame,
    price_col: str = "close",
    zscore_threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute both rolling and OU-KF Z-scores and generate signals.

    Parameters
    ----------
    df : pd.DataFrame
        Output from ``price_collector.PriceDataCollector.collect()``.
    price_col : str
    zscore_threshold : float
        Absolute Z-score above which a mean-reversion signal is triggered.

    Returns
    -------
    pd.DataFrame
        Input frame augmented with:
        - ``zscore_20``, ``zscore_50``, ``zscore_200``
        - ``ou_kf_mean``, ``ou_kf_std``, ``ou_kf_zscore``
        - ``mr_signal``: one of ``"LONG"``, ``"SHORT"``, ``"NEUTRAL"``
    """
    out = rolling_zscore(df, price_col=price_col)
    out = ou_kf_zscore(out, price_col=price_col)

    # Signal logic: use OU-KF Z-score as primary, rolling z20 as confirmation
    z_kf = out["ou_kf_zscore"]
    z_20 = out["zscore_20"]

    out["mr_signal"] = "NEUTRAL"
    out.loc[(z_kf < -zscore_threshold) & (z_20 < -1.0), "mr_signal"] = "LONG"
    out.loc[(z_kf > zscore_threshold) & (z_20 > 1.0), "mr_signal"] = "SHORT"

    return out
