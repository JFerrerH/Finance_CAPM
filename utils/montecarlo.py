import numpy as np
import pandas as pd


def run_monte_carlo(returns, current_price, n_simulations=500, n_months=24, seed=42):
    """
    Simulate future monthly price paths using Geometric Brownian Motion.

    Parameters
    ----------
    returns       : pd.Series of historical monthly returns (decimal)
    current_price : float, last closing price (t=0)
    n_simulations : int, number of simulation paths
    n_months      : int, number of future months to project
    seed          : int, random seed for reproducibility

    Returns
    -------
    simulated    : np.ndarray (n_months+1, n_simulations) — rows = time, cols = paths
    pct_paths    : dict {5, 25, 50, 75, 95} → np.ndarray of percentile paths
    final_prices : np.ndarray of terminal prices (last row of simulated)
    """
    np.random.seed(seed)
    mu    = float(returns.mean())
    sigma = float(returns.std())

    # GBM log-normal increments: S(t+dt) = S(t) * exp((μ - ½σ²)dt + σ√dt · Z)
    shocks  = np.random.normal(mu - 0.5 * sigma ** 2, sigma, (n_months, n_simulations))
    log_cum = np.cumsum(shocks, axis=0)
    paths   = current_price * np.exp(log_cum)

    # Prepend t=0 row (all simulations start at current_price)
    simulated = np.vstack([np.full((1, n_simulations), current_price), paths])

    pct_paths = {
        pct: np.percentile(simulated, pct, axis=1)
        for pct in [5, 25, 50, 75, 95]
    }
    final_prices = simulated[-1, :]

    return simulated, pct_paths, final_prices
