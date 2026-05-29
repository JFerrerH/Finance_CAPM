import numpy as np
import pandas as pd


def calculate_performance_metrics(data, Rf, beta):
    stock_returns = data["Monthly_Return_Stock"].dropna()
    monthly_rf = Rf / 12

    # Annualised return (geometric)
    ann_return = float((1 + stock_returns.mean()) ** 12 - 1)

    # Annualised volatility
    ann_vol = float(stock_returns.std() * np.sqrt(12))

    # Sharpe ratio
    sharpe = (ann_return - Rf) / ann_vol if ann_vol > 0 else None

    # Treynor ratio
    treynor = (ann_return - Rf) / float(beta) if beta and float(beta) != 0 else None

    # Sortino ratio — downside deviation only
    downside = stock_returns[stock_returns < monthly_rf]
    down_vol = float(downside.std() * np.sqrt(12)) if len(downside) > 1 else None
    sortino = (ann_return - Rf) / down_vol if down_vol and down_vol > 0 else None

    # Drawdown series
    cumulative  = (1 + stock_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else None

    return {
        "ann_return":      ann_return,
        "ann_vol":         ann_vol,
        "sharpe":          sharpe,
        "treynor":         treynor,
        "sortino":         sortino,
        "max_drawdown":    max_drawdown,
        "calmar":          calmar,
        "drawdown_series": drawdown,
        "cumulative":      cumulative,
    }


def calculate_rolling_beta(data, window=12):
    returns = data[["Monthly_Return_Stock", "Monthly_Return_Index"]].dropna()
    betas, dates = [], []
    for i in range(window, len(returns) + 1):
        chunk = returns.iloc[i - window:i]
        cov = np.cov(chunk["Monthly_Return_Index"], chunk["Monthly_Return_Stock"])
        b = cov[0, 1] / cov[0, 0] if cov[0, 0] != 0 else np.nan
        betas.append(b)
        dates.append(chunk.index[-1])
    return pd.Series(betas, index=dates, name="Rolling Beta")


def calculate_var_cvar(returns, confidence_levels=(0.95, 0.99)):
    """
    Historical Value at Risk and Conditional VaR (Expected Shortfall).
    Values are expressed as monthly returns — negative means loss.
    e.g. var_95 = -0.035 means: at 95% confidence, monthly loss won't exceed 3.5%.
    """
    results = {}
    for cl in confidence_levels:
        cl_pct = int(cl * 100)
        var  = float(np.percentile(returns, (1 - cl) * 100))
        tail = returns[returns <= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        results[f"var_{cl_pct}"]  = var
        results[f"cvar_{cl_pct}"] = cvar
    return results
