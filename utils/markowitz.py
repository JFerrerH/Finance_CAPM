import numpy as np
import pandas as pd


def run_efficient_frontier(mean_returns, cov_matrix, Rf, n_simulations=5000, seed=42):
    """
    Monte Carlo efficient frontier — vectorised GBM-free version.

    Generates `n_simulations` random weight vectors (uniform on the simplex)
    and computes annualised expected return, volatility, and Sharpe for each.

    Returns
    -------
    pd.DataFrame with columns: Return, Volatility, Sharpe, <ticker…>
    """
    np.random.seed(seed)
    n   = len(mean_returns)
    mu  = mean_returns.values          # (n,)
    cov = cov_matrix.values            # (n, n)

    # Random weights uniform on the simplex via Dirichlet(1,…,1)
    W = np.random.dirichlet(np.ones(n), n_simulations)   # (n_sim, n)

    # Annualised return (geometric) and volatility
    ret = (1.0 + W @ mu) ** 12 - 1.0
    var = np.einsum("ij,jk,ik->i", W, cov, W) * 12      # batch quadratic form
    vol = np.sqrt(np.maximum(var, 0.0))
    sr  = np.where(vol > 0, (ret - Rf) / vol, 0.0)

    data = np.column_stack([ret, vol, sr, W])
    cols = ["Return", "Volatility", "Sharpe"] + list(mean_returns.index)
    return pd.DataFrame(data, columns=cols)


def get_optimal_portfolios(frontier_df):
    """Return the max-Sharpe and min-Volatility rows from the frontier DataFrame."""
    max_sharpe = frontier_df.loc[frontier_df["Sharpe"].idxmax()].copy()
    min_vol    = frontier_df.loc[frontier_df["Volatility"].idxmin()].copy()
    return {"max_sharpe": max_sharpe, "min_vol": min_vol}
