import numpy as np
import plotly.graph_objects as go


def plot_sml(Rf, Rm, beta, capm_return, ticker):
    betas = np.linspace(0, 4, 20)
    expected_returns = Rf * 100 + betas * (Rm * 100 - Rf * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=betas, y=expected_returns,
        mode="lines", name="Security Market Line (SML)",
        line=dict(color="#3b82f6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[Rf * 100],
        mode="markers", name="Risk-Free Rate",
        marker=dict(color="#ef4444", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=[1], y=[Rm * 100],
        mode="markers", name="Market Return",
        marker=dict(color="#92400e", size=10),
    ))
    fig.add_trace(go.Scatter(
        x=[beta], y=[capm_return],
        mode="markers", name=f"{ticker} (β={beta:.2f})",
        marker=dict(color="#10b981", size=14, symbol="circle",
                    line=dict(color="white", width=2)),
    ))
    fig.update_layout(
        title=dict(text="Security Market Line (SML)", font=dict(size=16)),
        xaxis_title="Beta (β)",
        yaxis_title="Expected Return (%)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return fig


def plot_monthly_returns(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Monthly_Return_Stock"],
        mode="lines", name="Stock",
        line=dict(width=1.5, color="#3b82f6"),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Monthly_Return_Index"],
        mode="lines", name="Index",
        line=dict(width=1.5, color="#f59e0b"),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
    ))
    fig.update_layout(
        title=dict(text="Monthly Returns — Stock vs. Index", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Monthly Return",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return fig


def plot_regression(X, Y, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X["Monthly_Return_Index"], y=Y,
        mode="markers", name="Monthly observations",
        marker=dict(color="#3b82f6", opacity=0.55, size=7),
    ))
    sorted_idx = X["Monthly_Return_Index"].argsort()
    fig.add_trace(go.Scatter(
        x=X["Monthly_Return_Index"].iloc[sorted_idx],
        y=y_pred.iloc[sorted_idx],
        mode="lines", name="Regression line",
        line=dict(color="#ef4444", width=2),
    ))
    fig.update_layout(
        title=dict(text="Beta Estimation — OLS Regression", font=dict(size=16)),
        xaxis_title="Market Monthly Return",
        yaxis_title="Stock Monthly Return",
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return fig


def plot_valuation_comparison(current_price, pe_price, ddm_price, dcf_price, lynch_price):
    entries = [
        ("Current Price", current_price),
        ("P/E Method",    pe_price),
        ("DCF Method",    dcf_price),
        ("DDM Method",    ddm_price),
        ("Peter Lynch",   lynch_price),
    ]

    labels = [lbl for lbl, val in entries if val is not None]
    values = [val for _, val in entries if val is not None]

    if not values:
        return go.Figure()

    colors = []
    for lbl, val in entries:
        if val is None:
            continue
        if lbl == "Current Price":
            colors.append("#6366f1")
        elif current_price and val >= current_price:
            colors.append("#10b981")   # green  → stock looks undervalued
        else:
            colors.append("#ef4444")   # red    → stock looks overvalued

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"${v:,.2f}" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))

    if current_price:
        fig.add_vline(
            x=current_price,
            line_dash="dot",
            line_color="#6366f1",
            line_width=1.5,
            annotation_text=f"  Current  ${current_price:.2f}",
            annotation_position="top right",
            annotation_font_color="#6366f1",
        )

    max_val = max(values) * 1.25 if values else 1
    fig.update_layout(
        title=dict(text="Fair Value vs. Current Price", font=dict(size=16)),
        xaxis=dict(title="Price (USD)", range=[0, max_val]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        showlegend=False,
        margin=dict(t=50, l=10, r=40, b=40),
    )
    return fig


def plot_rolling_beta(rolling_beta, static_beta):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_beta.index, y=rolling_beta.values,
        mode="lines", name="12-Month Rolling β",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
    ))
    fig.add_hline(
        y=1.0, line_dash="dot", line_color="#94a3b8", line_width=1.5,
        annotation_text="  Market β = 1", annotation_position="top right",
        annotation_font_color="#94a3b8",
    )
    fig.add_hline(
        y=float(static_beta), line_dash="dash", line_color="#f59e0b", line_width=1.5,
        annotation_text=f"  Static β = {static_beta:.2f}", annotation_position="bottom right",
        annotation_font_color="#f59e0b",
    )
    fig.update_layout(
        title=dict(text="Rolling 12-Month Beta", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Beta (β)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=120),
    )
    return fig


def plot_drawdown(drawdown_series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index, y=drawdown_series.values * 100,
        mode="lines", name="Drawdown",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
    ))
    fig.update_layout(
        title=dict(text="Underwater Chart (Drawdown %)", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=60),
    )
    return fig


def plot_return_distribution(returns):
    import numpy as np
    from scipy import stats

    mean = float(returns.mean())
    std  = float(returns.std())
    skew = float(returns.skew())
    kurt = float(returns.kurt())    # excess kurtosis
    x    = np.linspace(returns.min(), returns.max(), 300)
    normal_curve = stats.norm.pdf(x, mean, std)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        name="Monthly Returns",
        marker_color="#3b82f6",
        opacity=0.65,
        histnorm="probability density",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=normal_curve,
        mode="lines", name="Normal fit",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    ))
    fig.add_vline(x=0,    line_dash="dot",   line_color="#94a3b8", line_width=1.5)
    fig.add_vline(x=mean, line_dash="solid", line_color="#10b981", line_width=1.5,
                  annotation_text=f"  Mean {mean:.2%}", annotation_position="top right",
                  annotation_font_color="#10b981")
    fig.update_layout(
        title=dict(
            text=f"Return Distribution  |  Skew: {skew:.2f}  |  Excess Kurtosis: {kurt:.2f}",
            font=dict(size=15),
        ),
        xaxis_title="Monthly Return",
        yaxis_title="Density",
        hovermode="x unified",
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return fig


def plot_var_distribution(returns, var_results):
    """Histogram of monthly returns with VaR/CVaR threshold lines."""
    import numpy as np
    from scipy import stats

    mean = float(returns.mean())
    std  = float(returns.std())
    x    = np.linspace(returns.min(), returns.max(), 300)
    normal_curve = stats.norm.pdf(x, mean, std)

    var_95  = var_results["var_95"]
    cvar_95 = var_results["cvar_95"]
    var_99  = var_results["var_99"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=30,
        name="Monthly Returns",
        marker_color="#3b82f6",
        opacity=0.60,
        histnorm="probability density",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=normal_curve,
        mode="lines", name="Normal fit",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    ))
    fig.add_vline(x=var_95,  line_dash="solid", line_color="#ef4444", line_width=2,
                  annotation_text=f"VaR 95%: {var_95:.2%}", annotation_position="top left",
                  annotation_font_color="#ef4444")
    fig.add_vline(x=var_99,  line_dash="solid", line_color="#7f1d1d", line_width=2,
                  annotation_text=f"VaR 99%: {var_99:.2%}", annotation_position="bottom left",
                  annotation_font_color="#991b1b")
    fig.add_vline(x=cvar_95, line_dash="dot",   line_color="#f97316", line_width=1.5,
                  annotation_text=f"CVaR 95%: {cvar_95:.2%}", annotation_position="top right",
                  annotation_font_color="#f97316")
    fig.update_layout(
        title=dict(text="Monthly Return Distribution with VaR Thresholds", font=dict(size=16)),
        xaxis_title="Monthly Return",
        yaxis_title="Density",
        hovermode="x unified",
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return fig


def plot_monte_carlo(pct_paths, future_dates, current_price, ticker):
    """Fan chart of Monte Carlo GBM percentile paths."""
    dates_list = list(future_dates)
    dates_rev  = list(future_dates[::-1])

    fig = go.Figure()

    # 90% confidence band (5th – 95th)
    fig.add_trace(go.Scatter(
        x=dates_list + dates_rev,
        y=list(pct_paths[95]) + list(pct_paths[5][::-1]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.10)",
        line=dict(width=0),
        name="90% CI",
        hoverinfo="skip",
    ))
    # 50% confidence band (25th – 75th)
    fig.add_trace(go.Scatter(
        x=dates_list + dates_rev,
        y=list(pct_paths[75]) + list(pct_paths[25][::-1]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.22)",
        line=dict(width=0),
        name="50% CI",
        hoverinfo="skip",
    ))
    # Percentile boundary lines
    fig.add_trace(go.Scatter(
        x=dates_list, y=pct_paths[5],
        mode="lines", name="5th pct",
        line=dict(color="#ef4444", width=1.2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=dates_list, y=pct_paths[95],
        mode="lines", name="95th pct",
        line=dict(color="#10b981", width=1.2, dash="dot"),
    ))
    # Median path
    fig.add_trace(go.Scatter(
        x=dates_list, y=pct_paths[50],
        mode="lines", name="Median path",
        line=dict(color="#3b82f6", width=2.5),
    ))
    # Current price reference line
    fig.add_hline(
        y=current_price, line_dash="dash", line_color="#94a3b8", line_width=1.5,
        annotation_text=f"  Current ${current_price:.2f}",
        annotation_position="top right",
        annotation_font_color="#94a3b8",
    )
    fig.update_layout(
        title=dict(text=f"Monte Carlo Price Simulation — {ticker.upper()}", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=120),
    )
    return fig


def plot_terminal_distribution(final_prices, current_price):
    """Histogram of Monte Carlo terminal prices with key reference lines."""
    import numpy as np

    mean_p = float(np.mean(final_prices))
    p10    = float(np.percentile(final_prices, 10))
    p90    = float(np.percentile(final_prices, 90))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_prices, nbinsx=50,
        name="Terminal Prices",
        marker_color="#3b82f6",
        opacity=0.70,
    ))
    fig.add_vline(x=current_price, line_dash="dash",  line_color="#94a3b8", line_width=2,
                  annotation_text=f"  Current ${current_price:.2f}",
                  annotation_position="top right", annotation_font_color="#94a3b8")
    fig.add_vline(x=mean_p,        line_dash="solid", line_color="#10b981", line_width=2,
                  annotation_text=f"  Mean ${mean_p:.2f}",
                  annotation_position="top right", annotation_font_color="#10b981")
    fig.add_vline(x=p10,           line_dash="dot",   line_color="#ef4444", line_width=1.5,
                  annotation_text=f"  10th pct ${p10:.2f}",
                  annotation_position="top left",  annotation_font_color="#ef4444")
    fig.add_vline(x=p90,           line_dash="dot",   line_color="#10b981", line_width=1.5,
                  annotation_text=f"  90th pct ${p90:.2f}",
                  annotation_position="top right", annotation_font_color="#10b981")
    fig.update_layout(
        title=dict(text="Terminal Price Distribution (Monte Carlo)", font=dict(size=16)),
        xaxis_title="Price (USD)",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=60),
    )
    return fig
