import numpy as np
import statsmodels.api as sm


def calculate_capm(data, Rf):
    correlation = data["Monthly_Return_Stock"].corr(data["Monthly_Return_Index"])
    X = sm.add_constant(data["Monthly_Return_Index"])
    Y = data["Monthly_Return_Stock"]
    lm = sm.OLS(Y, X).fit()
    intercept, beta = lm.params
    y_pred = beta * X["Monthly_Return_Index"] + intercept
    Rm = ((1 + data["Monthly_Return_Index"].mean()) ** 12) - 1
    capm_return = (Rf * 100) + (beta * ((Rm - Rf) * 100))

    # OLS diagnostics
    r_squared = float(lm.rsquared)
    alpha_pvalue = float(lm.pvalues.iloc[0])
    beta_pvalue  = float(lm.pvalues.iloc[1])

    # Annualised Jensen's Alpha (compound)
    jensen_alpha_annual = float((1 + float(intercept)) ** 12 - 1)

    # Risk decomposition: systematic vs unsystematic
    market_var      = float(data["Monthly_Return_Index"].var())
    total_var       = float(data["Monthly_Return_Stock"].var())
    systematic_var  = (float(beta) ** 2) * market_var
    unsystematic_var = max(total_var - systematic_var, 0.0)
    systematic_pct  = systematic_var / total_var if total_var > 0 else 0.0

    return {
        "beta": beta,
        "intercept": intercept,
        "Rm": Rm,
        "capm_return": capm_return,
        "correlation": correlation,
        "X": X,
        "Y": Y,
        "y_pred": y_pred,
        # Diagnostics
        "r_squared": r_squared,
        "alpha_pvalue": alpha_pvalue,
        "beta_pvalue": beta_pvalue,
        "jensen_alpha_annual": jensen_alpha_annual,
        "systematic_pct": systematic_pct,
        "unsystematic_pct": 1.0 - systematic_pct,
    }
