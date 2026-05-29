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

    return {
        "beta": beta,
        "intercept": intercept,
        "Rm": Rm,
        "capm_return": capm_return,
        "correlation": correlation,
        "X": X,
        "Y": Y,
        "y_pred": y_pred,
    }
