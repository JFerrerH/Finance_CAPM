import numpy as np
import pandas as pd
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


# ─────────────────────────────────────────────────────────────────────────────
# Markowitz / Portfolio charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(returns_df):
    """Diverging heatmap of the pairwise correlation matrix."""
    corr   = returns_df.corr().round(4)
    labels = corr.columns.tolist()
    n      = len(labels)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=13),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Corr", thickness=14),
    ))
    fig.update_layout(
        title=dict(text="Asset Correlation Matrix", font=dict(size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(320, 80 * n),
        margin=dict(t=60, b=40, l=60, r=40),
    )
    return fig


def plot_efficient_frontier(frontier_df, max_sharpe, min_vol, Rf, asset_stats=None, tickers=None):
    """
    Efficient frontier scatter coloured by Sharpe ratio.

    Parameters
    ----------
    asset_stats : list of dicts with keys: name, return_ann, vol_ann
    tickers     : list of ticker strings — used to embed weights in customdata
    """
    weight_cols = tickers if tickers else [
        c for c in frontier_df.columns if c not in ("Return", "Volatility", "Sharpe")
    ]
    customdata = frontier_df[weight_cols].values  # shape (n_sim, n_tickers)

    # Build per-ticker weight lines for the hover tooltip
    weight_hover = "<br>".join(
        f"{t}: %{{customdata[{i}]:.1%}}" for i, t in enumerate(weight_cols)
    )

    fig = go.Figure()

    # ── All random portfolios (coloured by Sharpe) ──
    fig.add_trace(go.Scatter(
        x=frontier_df["Volatility"] * 100,
        y=frontier_df["Return"] * 100,
        mode="markers",
        customdata=customdata,
        marker=dict(
            color=frontier_df["Sharpe"],
            colorscale="Viridis",
            size=5,
            opacity=0.55,
            showscale=True,
            colorbar=dict(title="Sharpe", thickness=14, x=1.02),
        ),
        name="Random Portfolios",
        hovertemplate=(
            "<b>Portfolio</b><br>"
            "Vol: %{x:.2f}%   Return: %{y:.2f}%   Sharpe: %{marker.color:.2f}"
            f"<br><br><b>Weights</b><br>{weight_hover}"
            "<extra></extra>"
        ),
    ))

    # ── Capital Market Line ──
    ms_vol = float(max_sharpe["Volatility"])
    ms_ret = float(max_sharpe["Return"])
    if ms_vol > 0:
        cml_x = np.linspace(0, ms_vol * 1.7, 60)
        cml_y = Rf + (ms_ret - Rf) / ms_vol * cml_x
        fig.add_trace(go.Scatter(
            x=cml_x * 100, y=cml_y * 100,
            mode="lines", name="Capital Market Line",
            line=dict(color="#10b981", width=1.8, dash="dash"),
        ))

    # ── Individual assets ──
    if asset_stats:
        for a in asset_stats:
            fig.add_trace(go.Scatter(
                x=[a["vol_ann"] * 100], y=[a["return_ann"] * 100],
                mode="markers+text",
                marker=dict(color="#94a3b8", size=11, symbol="circle",
                            line=dict(color="white", width=1.5)),
                text=[a["name"]],
                textposition="top center",
                name=a["name"],
                hovertemplate=f"<b>{a['name']}</b><br>Vol: {a['vol_ann']*100:.2f}%<br>"
                              f"Return: {a['return_ann']*100:.2f}%<extra></extra>",
            ))

    # ── Min Vol portfolio ──
    fig.add_trace(go.Scatter(
        x=[float(min_vol["Volatility"]) * 100],
        y=[float(min_vol["Return"]) * 100],
        mode="markers",
        marker=dict(color="#3b82f6", size=16, symbol="diamond",
                    line=dict(color="white", width=2)),
        name=f"Min Vol  (SR={float(min_vol['Sharpe']):.2f})",
        hovertemplate=f"Min Vol<br>Vol: {float(min_vol['Volatility'])*100:.2f}%<br>"
                      f"Return: {float(min_vol['Return'])*100:.2f}%<extra></extra>",
    ))

    # ── Max Sharpe portfolio ──
    fig.add_trace(go.Scatter(
        x=[ms_vol * 100], y=[ms_ret * 100],
        mode="markers",
        marker=dict(color="#f59e0b", size=18, symbol="star",
                    line=dict(color="white", width=2)),
        name=f"Max Sharpe  (SR={float(max_sharpe['Sharpe']):.2f})",
        hovertemplate=f"Max Sharpe<br>Vol: {ms_vol*100:.2f}%<br>"
                      f"Return: {ms_ret*100:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Efficient Frontier (Monte Carlo — 5 000 portfolios)", font=dict(size=16)),
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, r=60),
    )
    return fig


def plot_portfolio_weights(tickers, ms_weights, mv_weights):
    """Grouped bar chart comparing Max-Sharpe and Min-Vol portfolio weights."""
    pct_ms = np.array(ms_weights) * 100
    pct_mv = np.array(mv_weights) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Max Sharpe ⭐",
        x=tickers, y=pct_ms,
        marker_color="#f59e0b",
        text=[f"{w:.1f}%" for w in pct_ms],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Min Volatility 💎",
        x=tickers, y=pct_mv,
        marker_color="#3b82f6",
        text=[f"{w:.1f}%" for w in pct_mv],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Optimal Portfolio Weights", font=dict(size=16)),
        xaxis_title="Asset",
        yaxis_title="Weight (%)",
        yaxis=dict(range=[0, max(pct_ms.max(), pct_mv.max()) * 1.25]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return fig


def plot_weights_pie(tickers, weights, title="Selected Portfolio"):
    """Donut chart for a single portfolio's weight breakdown."""
    # Filter out near-zero weights for readability
    pairs = [(t, float(w)) for t, w in zip(tickers, weights) if float(w) > 0.001]
    labels, values = zip(*pairs) if pairs else (tickers, list(weights))

    fig = go.Figure(go.Pie(
        labels=list(labels),
        values=[round(v * 100, 2) for v in values],
        hole=0.45,
        texttemplate="<b>%{label}</b><br>%{value:.1f}%",
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>",
        marker=dict(line=dict(color="rgba(255,255,255,0.15)", width=2)),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5),
        height=300,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Financial Health charts
# ─────────────────────────────────────────────────────────────────────────────

def _B(v):
    """Scale raw USD value to billions for chart axes."""
    return v / 1e9 if v is not None else 0.0


def plot_revenue_earnings_trend(income_data: dict) -> go.Figure:
    """
    Grouped bar chart: Revenue vs Net Income over up to 4 years.
    Secondary y-axis shows Net Margin %.
    """
    years    = income_data.get("years", [])
    revenue  = income_data.get("revenue") or []
    net_inc  = income_data.get("net_income") or []

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Revenue",
        x=years, y=[_B(v) for v in revenue],
        marker_color="#3b82f6",
        text=[f"${_B(v):.1f}B" for v in revenue],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Net Income",
        x=years, y=[_B(v) for v in net_inc],
        marker_color="#10b981",
        text=[f"${_B(v):.1f}B" for v in net_inc],
        textposition="outside",
    ))
    # Net margin line
    margins = []
    for r, n in zip(revenue, net_inc):
        margins.append(round(n / r * 100, 1) if r and r != 0 else 0)
    fig.add_trace(go.Scatter(
        name="Net Margin %",
        x=years, y=margins,
        mode="lines+markers+text",
        text=[f"{m:.1f}%" for m in margins],
        textposition="top center",
        line=dict(color="#f59e0b", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Revenue & Net Income", font=dict(size=16)),
        xaxis_title="Year",
        yaxis=dict(title="USD Billions", tickprefix="$", ticksuffix="B"),
        yaxis2=dict(title="Net Margin (%)", overlaying="y", side="right",
                    ticksuffix="%", showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def plot_sankey(income_data: dict, ticker: str = "") -> go.Figure:
    """
    Sankey diagram for the most recent fiscal year P&L flow:
    Revenue → COGS / Gross Profit → OpEx / EBIT → Tax+Interest / Net Income
    """
    def _last(key):
        vals = income_data.get(key)
        return float(vals[-1]) if vals else 0.0

    revenue   = _last("revenue")
    cogs      = _last("cost_of_revenue")
    gross     = _last("gross_profit")
    op_inc    = _last("operating_income")
    net_inc   = _last("net_income")
    tax       = abs(_last("tax"))
    year      = income_data.get("years", [""])[-1]

    # If gross profit not directly available, derive it
    if gross == 0 and revenue and cogs:
        gross = revenue - cogs
    if cogs == 0 and revenue and gross:
        cogs = revenue - gross

    opex        = max(gross - op_inc, 0)        # SG&A + R&D etc.
    non_op_tax  = max(op_inc - net_inc, 0)      # interest + tax burden

    def _fmt(v):
        b = abs(v) / 1e9
        return f"${b:.2f}B" if b >= 1 else f"${abs(v)/1e6:.0f}M"

    labels = [
        f"Revenue<br>{_fmt(revenue)}",           # 0
        f"Cost of Revenue<br>{_fmt(cogs)}",      # 1
        f"Gross Profit<br>{_fmt(gross)}",        # 2
        f"Op. Expenses<br>{_fmt(opex)}",         # 3
        f"EBIT<br>{_fmt(op_inc)}",               # 4
        f"Tax & Interest<br>{_fmt(non_op_tax)}", # 5
        f"Net Income<br>{_fmt(net_inc)}",        # 6
    ]

    node_colors = [
        "#3b82f6",  # Revenue — blue
        "#ef4444",  # COGS — red
        "#8b5cf6",  # Gross Profit — purple
        "#f97316",  # OpEx — orange
        "#06b6d4",  # EBIT — cyan
        "#ef4444",  # Tax & Interest — red
        "#10b981",  # Net Income — green
    ]

    sources = [0, 0, 2, 2, 4, 4]
    targets = [1, 2, 3, 4, 5, 6]
    values  = [
        max(cogs, 1),
        max(gross, 1),
        max(opex, 1),
        max(op_inc, 1),
        max(non_op_tax, 1),
        max(net_inc, 1),
    ]

    link_colors = [
        "rgba(239,68,68,0.35)",   # COGS
        "rgba(139,92,246,0.35)",  # Gross Profit
        "rgba(249,115,22,0.35)",  # OpEx
        "rgba(6,182,212,0.35)",   # EBIT
        "rgba(239,68,68,0.35)",   # Tax & Interest
        "rgba(16,185,129,0.45)",  # Net Income
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=22,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))
    fig.update_layout(
        title=dict(
            text=f"P&L Money Flow — {ticker} ({year})",
            font=dict(size=16),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        height=420,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def plot_cashflow_bars(cf_data: dict) -> go.Figure:
    """
    Grouped bar chart: Operating CF, Free CF, Investing CF, Financing CF by year.
    """
    years     = cf_data.get("years", [])
    operating = cf_data.get("operating") or []
    fcf       = cf_data.get("fcf") or []
    investing = cf_data.get("investing") or []
    financing = cf_data.get("financing") or []

    def _b(lst):
        return [_B(v) for v in lst]

    fig = go.Figure()
    if operating:
        fig.add_trace(go.Bar(name="Operating CF",  x=years, y=_b(operating),
                             marker_color="#3b82f6"))
    if fcf:
        fig.add_trace(go.Bar(name="Free CF",       x=years, y=_b(fcf),
                             marker_color="#10b981"))
    if investing:
        fig.add_trace(go.Bar(name="Investing CF",  x=years, y=_b(investing),
                             marker_color="#f59e0b"))
    if financing:
        fig.add_trace(go.Bar(name="Financing CF",  x=years, y=_b(financing),
                             marker_color="#8b5cf6"))

    fig.update_layout(
        barmode="group",
        title=dict(text="Cash Flow Statement", font=dict(size=16)),
        xaxis_title="Year",
        yaxis=dict(title="USD Billions", tickprefix="$", ticksuffix="B"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


def plot_capital_structure(bal_data: dict) -> go.Figure:
    """
    Stacked bar: Total Debt vs Stockholders Equity over years.
    Line overlay: D/E ratio on secondary axis.
    """
    years  = bal_data.get("years", [])
    debt   = bal_data.get("total_debt")  or []
    equity = bal_data.get("equity")      or []

    de_ratio = []
    for d, e in zip(debt, equity):
        de_ratio.append(round(d / e, 2) if e and e != 0 else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Total Debt", x=years, y=[_B(v) for v in debt],
        marker_color="#ef4444",
        text=[f"${_B(v):.1f}B" for v in debt], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Stockholders Equity", x=years, y=[_B(v) for v in equity],
        marker_color="#10b981",
        text=[f"${_B(v):.1f}B" for v in equity], textposition="inside",
    ))
    fig.add_trace(go.Scatter(
        name="D/E Ratio",
        x=years, y=de_ratio,
        mode="lines+markers+text",
        text=[f"{r:.2f}x" for r in de_ratio],
        textposition="top center",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Capital Structure", font=dict(size=16)),
        xaxis_title="Year",
        yaxis=dict(title="USD Billions", tickprefix="$", ticksuffix="B"),
        yaxis2=dict(title="D/E Ratio", overlaying="y", side="right",
                    ticksuffix="x", showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Macro / Dalio framework charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_dalio_quadrant(growth_score: float, inflation_score: float,
                        phase: str, phase_color: str) -> go.Figure:
    """
    2×2 scatter quadrant (x=growth, y=inflation).
    Background zones coloured per Dalio's four environments.
    A star marks the current estimated position.
    """
    fig = go.Figure()

    # Background rectangles for each quadrant
    zones = [
        # x0, y0, x1, y1, fill,                          label
        (-1.1, 0,    0,    1.1, "rgba(99,102,241,0.12)",  "DEFLATIONARY BUST"),
        (0,    0,    1.1,  1.1, "rgba(245,158,11,0.12)",  "INFLATIONARY BOOM"),
        (-1.1, -1.1, 0,    0,   "rgba(16,185,129,0.12)",  "GOLDILOCKS"),
        (0,   -1.1,  1.1,  0,   "rgba(239,68,68,0.12)",   "STAGFLATION"),
    ]
    for x0, y0, x1, y1, fill, label in zones:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=fill, line_width=0, layer="below")
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=10, color="rgba(200,200,200,0.55)"),
        )

    # Axis dividers
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.35)", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(150,150,150,0.35)", line_width=1)

    # Current position
    fig.add_trace(go.Scatter(
        x=[growth_score], y=[inflation_score],
        mode="markers",
        marker=dict(symbol="star", size=26, color=phase_color,
                    line=dict(color="white", width=2)),
        name=phase,
        hovertemplate=(
            f"<b>{phase}</b><br>"
            f"Growth signal: {growth_score:+.2f}<br>"
            f"Inflation signal: {inflation_score:+.2f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="Economic Cycle Positioning — Dalio Framework", font=dict(size=16)),
        xaxis=dict(
            title="← Contracting  |  Growth  |  Expanding →",
            range=[-1.15, 1.15], zeroline=False, showgrid=False,
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        yaxis=dict(
            title="← Deflationary  |  Inflation  |  Elevated →",
            range=[-1.15, 1.15], zeroline=False, showgrid=False,
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=400,
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def plot_macro_history(macro_data: dict, names: list[str]) -> go.Figure:
    """
    Normalised index chart (base = 100 at start) for selected macro series.
    Lets the user compare trends across indicators.
    """
    COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
    fig = go.Figure()

    for i, name in enumerate(names):
        s = macro_data.get(name)
        if s is None or s.empty:
            continue
        base = float(s.iloc[0])
        if base == 0:
            continue
        normalised = s / base * 100
        fig.add_trace(go.Scatter(
            x=normalised.index, y=normalised.values,
            mode="lines", name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Macro Indicators — Normalised (base=100)", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="Index (base = 100 at start)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    return fig


def plot_short_cycle_path(path_df, current_growth: float, current_inflation: float,
                          cycle_phase: str, cycle_color: str,
                          sp_series=None) -> go.Figure:
    """
    Dalio short-term debt cycle visualization.

    Primary (sp_series provided): S&P 500 indexed to 100 oscillating above/below
    its long-term log-linear productivity trend — matching Dalio's 'How the Economic
    Machine Works' diagram where short cycles are waves riding above/below the
    rising productivity line.  Green fill = expansion above trend; red = contraction.

    Fallback (no sp_series): growth + inflation signals as two time-series lines.
    """
    PHASE_COLORS = {
        "Goldilocks":        "rgba(16,185,129,0.12)",
        "Inflationary Boom": "rgba(245,158,11,0.12)",
        "Stagflation":       "rgba(239,68,68,0.12)",
        "Deflationary Bust": "rgba(99,102,241,0.12)",
    }

    def _phase_at(g, infl):
        if g > 0 and infl <= 0:  return "Goldilocks"
        if g > 0 and infl >  0:  return "Inflationary Boom"
        if g <= 0 and infl > 0:  return "Stagflation"
        return "Deflationary Bust"

    fig = go.Figure()
    if path_df.empty:
        return fig

    # ── Phase background spans ─────────────────────────────────────────────────
    pf_dates = path_df.index
    phases   = [_phase_at(path_df["growth"].iloc[i], path_df["inflation"].iloc[i])
                for i in range(len(path_df))]
    runs: list[tuple] = []
    run_start, cur = 0, phases[0]
    for i in range(1, len(phases)):
        if phases[i] != cur:
            runs.append((run_start, i - 1, cur))
            run_start, cur = i, phases[i]
    runs.append((run_start, len(phases) - 1, cur))

    labeled: set = set()
    for s_i, e_i, ph in runs:
        x0 = pf_dates[s_i].strftime("%Y-%m-%d")
        x1 = pf_dates[e_i].strftime("%Y-%m-%d")
        fig.add_shape(type="rect", xref="x", yref="paper",
                      x0=x0, x1=x1, y0=0, y1=1,
                      fillcolor=PHASE_COLORS[ph], line_width=0, layer="below")
        if ph not in labeled and (e_i - s_i) >= 3:
            mid = pf_dates[(s_i + e_i) // 2].strftime("%Y-%m-%d")
            fig.add_annotation(xref="x", yref="paper",
                               x=mid, y=0.97,
                               text=f"<b>{ph}</b>",
                               showarrow=False, xanchor="center",
                               font=dict(size=9, color="rgba(220,220,220,0.50)"))
            labeled.add(ph)

    # ── Primary: S&P 500 vs log-linear productivity trend ─────────────────────
    if sp_series is not None and len(sp_series) >= 24:
        window_start = pf_dates[0]
        sp_win = sp_series[sp_series.index >= window_start].dropna()

        if len(sp_win) >= 12:
            # Fit log-linear trend on the FULL sp_series (long-horizon structural trend)
            t_all    = np.arange(len(sp_series), dtype=float)
            log_all  = np.log(sp_series.values.astype(float))
            slope, intercept = np.polyfit(t_all, log_all, 1)

            # Map window dates to positions in the full series for trend evaluation
            pos   = np.clip(sp_series.index.searchsorted(sp_win.index), 0, len(sp_series) - 1)
            t_win = pos.astype(float)

            sp_vals    = sp_win.values.astype(float)
            trend_vals = np.exp(slope * t_win + intercept)

            # Index both to 100 at window start so they share a common baseline
            sp_idx    = sp_vals    / sp_vals[0]    * 100
            trend_idx = trend_vals / trend_vals[0] * 100

            dates_sp = sp_win.index

            # Bidirectional fill: green above trend (expansion), red below (contraction)
            # tonexty fills from the current trace DOWN to the previous trace.
            above_clip = np.where(sp_idx >= trend_idx, sp_idx, trend_idx)
            below_clip = np.where(sp_idx <= trend_idx, sp_idx, trend_idx)

            # Green band: trend (reference) → above_clip
            fig.add_trace(go.Scatter(
                x=list(dates_sp), y=list(trend_idx),
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=list(dates_sp), y=list(above_clip),
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(16,185,129,0.22)",
                showlegend=False, hoverinfo="skip",
            ))

            # Red band: below_clip (reference) → trend
            fig.add_trace(go.Scatter(
                x=list(dates_sp), y=list(below_clip),
                mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=list(dates_sp), y=list(trend_idx),
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(239,68,68,0.22)",
                showlegend=False, hoverinfo="skip",
            ))

            # Productivity trend line (structural)
            fig.add_trace(go.Scatter(
                x=list(dates_sp), y=list(trend_idx),
                mode="lines", name="Productivity Trend",
                line=dict(color="rgba(148,163,184,0.80)", width=1.8, dash="dash"),
                hovertemplate="<b>Trend</b>: %{y:.1f}<br>%{x|%b %Y}<extra></extra>",
            ))

            # S&P 500 oscillating curve
            fig.add_trace(go.Scatter(
                x=list(dates_sp), y=list(sp_idx),
                mode="lines", name="S&P 500",
                line=dict(color="#3b82f6", width=2.8),
                hovertemplate="<b>S&P 500</b>: %{y:.1f}<br>%{x|%b %Y}<extra></extra>",
            ))

            # NOW marker
            now_str = dates_sp[-1].strftime("%Y-%m-%d")
            now_y   = float(sp_idx[-1])
            fig.add_shape(type="line", xref="x", yref="paper",
                          x0=now_str, x1=now_str, y0=0, y1=1,
                          line=dict(color=cycle_color, width=1.5, dash="dot"))
            fig.add_trace(go.Scatter(
                x=[dates_sp[-1]], y=[now_y],
                mode="markers",
                marker=dict(symbol="circle", size=11, color=cycle_color,
                            line=dict(color="white", width=2)),
                name=f"NOW — {cycle_phase}",
                hovertemplate=(
                    f"<b>NOW — {cycle_phase}</b><br>"
                    "%{x|%b %Y}: %{y:.1f}<extra></extra>"
                ),
            ))
            fig.add_annotation(xref="x", yref="paper",
                               x=now_str, y=0.86,
                               text=f"  NOW<br><b>{cycle_phase}</b>",
                               showarrow=False, xanchor="left",
                               font=dict(color=cycle_color, size=10),
                               bgcolor="rgba(15,15,20,0.65)", borderpad=4)

            fig.update_layout(
                title=dict(
                    text="Short Cycle — S&P 500 Oscillating Around Productivity Trend (Dalio)",
                    font=dict(size=15),
                ),
                xaxis=dict(showgrid=False, showline=False),
                yaxis=dict(
                    title="Index (= 100 at window start)",
                    showgrid=True, gridcolor="rgba(150,150,150,0.07)",
                    zeroline=False,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0, font=dict(size=11)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                height=430,
                margin=dict(t=60, b=40, l=80, r=30),
            )
            return fig

    # ── Fallback: growth & inflation signal lines ──────────────────────────────
    dates  = path_df.index
    g_vals = path_df["growth"].values
    i_vals = path_df["inflation"].values

    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(200,200,200,0.35)", line_width=1)

    fig.add_trace(go.Scatter(
        x=dates, y=g_vals,
        mode="lines", name="Growth (S&P momentum)",
        line=dict(color="#3b82f6", width=2.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
        hovertemplate="<b>%{x|%b %Y}</b>  Growth: %{y:+.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=i_vals,
        mode="lines", name="Inflation (Gold + rates)",
        line=dict(color="#f59e0b", width=2.5),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.09)",
        hovertemplate="<b>%{x|%b %Y}</b>  Inflation: %{y:+.2f}<extra></extra>",
    ))

    now_str = dates[-1].strftime("%Y-%m-%d")
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=now_str, x1=now_str, y0=0, y1=1,
                  line=dict(color=cycle_color, width=2))
    fig.add_annotation(xref="x", yref="paper",
                       x=now_str, y=0.88,
                       text=(f"NOW<br><b>{cycle_phase}</b><br>"
                             f"<span style='font-size:9px'>"
                             f"G {current_growth:+.2f} / I {current_inflation:+.2f}</span>"),
                       showarrow=False, xanchor="left",
                       font=dict(color=cycle_color, size=10),
                       bgcolor="rgba(15,15,20,0.6)", borderpad=4)

    fig.update_layout(
        title=dict(text="Short Cycle — Growth & Inflation Over Time (Dalio Framework)",
                   font=dict(size=15)),
        xaxis=dict(showgrid=False, showline=False),
        yaxis=dict(
            title="Signal strength",
            range=[-1.15, 1.15],
            showgrid=True, gridcolor="rgba(150,150,150,0.07)",
            zeroline=False,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1  Contraction/Deflation", "-0.5", "0", "+0.5",
                      "+1  Expansion/Inflation"],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        height=430,
        margin=dict(t=60, b=40, l=80, r=30),
    )
    return fig


def plot_big_cycle_history(history: dict, bc_phase: str, bc_color: str,
                           bc_icon: str) -> go.Figure:
    """
    Long-term chart showing Federal Debt/GDP (%) from ~1966 to today,
    overlaid with the Federal Funds Rate.

    Horizontal threshold lines mark Dalio's structural danger zones.
    Key historical events are annotated.
    The current position is highlighted with a vertical 'NOW' marker.
    """
    debt_gdp  = history.get("Debt/GDP")
    fedfunds  = history.get("Fed Funds")

    if debt_gdp is None or debt_gdp.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Federal Debt / GDP — History (FRED data unavailable)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    fig = go.Figure()

    # ── Debt / GDP area ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=debt_gdp.index, y=debt_gdp.values,
        mode="lines", name="Federal Debt / GDP",
        line=dict(color="#6366f1", width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
        hovertemplate="<b>Debt/GDP</b>: %{y:.1f}%<br>%{x|%b %Y}<extra></extra>",
    ))

    # ── Danger-zone threshold lines ───────────────────────────────────────────
    for level, label, color in [
        (90,  "90% — Late Expansion threshold",   "rgba(245,158,11,0.6)"),
        (115, "115% — Extreme leverage",          "rgba(239,68,68,0.6)"),
    ]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1.5,
                      annotation_text=f"  {label}",
                      annotation_position="top right",
                      annotation_font=dict(color=color, size=10))

    # ── Fed Funds Rate (secondary axis) ───────────────────────────────────────
    if fedfunds is not None and not fedfunds.empty:
        fig.add_trace(go.Scatter(
            x=fedfunds.index, y=fedfunds.values,
            mode="lines", name="Fed Funds Rate",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="<b>Fed Funds</b>: %{y:.2f}%<br>%{x|%b %Y}<extra></extra>",
            opacity=0.75,
        ))

    # ── Key historical event annotations ─────────────────────────────────────
    events = [
        ("1971-08-15", "Nixon<br>gold window"),
        ("1981-06-01", "Volcker<br>peak rates"),
        ("2008-09-15", "GFC"),
        ("2020-03-01", "COVID"),
    ]
    # add_shape + add_annotation avoids add_vline's internal Timestamp arithmetic
    # which is incompatible with pandas 2.0 on datetime axes.
    for date_str, label in events:
        try:
            dt = pd.to_datetime(date_str)
            if debt_gdp.index.min() <= dt <= debt_gdp.index.max():
                x_str = dt.strftime("%Y-%m-%d")
                fig.add_shape(type="line", xref="x", yref="paper",
                              x0=x_str, x1=x_str, y0=0, y1=1,
                              line=dict(color="rgba(148,163,184,0.30)", dash="dot", width=1))
                fig.add_annotation(xref="x", yref="paper",
                                   x=x_str, y=0.97,
                                   text=f"  {label}", showarrow=False,
                                   font=dict(size=9, color="rgba(148,163,184,0.65)"),
                                   xanchor="left")
        except Exception:
            pass

    # ── Current position ──────────────────────────────────────────────────────
    now_str = debt_gdp.index.max().strftime("%Y-%m-%d")
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=now_str, x1=now_str, y0=0, y1=1,
                  line=dict(color=bc_color, dash="solid", width=2))
    fig.add_annotation(xref="x", yref="paper",
                       x=now_str, y=0.88,
                       text=f"  {bc_icon} NOW — {bc_phase}",
                       showarrow=False,
                       font=dict(color=bc_color, size=11),
                       xanchor="left")

    fig.update_layout(
        title=dict(text="Big Cycle — Federal Debt / GDP Since 1966 (Dalio Long-term Debt Cycle)",
                   font=dict(size=15)),
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="Federal Debt / GDP (%)", ticksuffix="%",
                   showgrid=True, gridcolor="rgba(150,150,150,0.1)"),
        yaxis2=dict(title="Fed Funds Rate (%)", overlaying="y", side="right",
                    ticksuffix="%", showgrid=False, range=[0, 25]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
        margin=dict(t=65, b=40, l=60, r=70),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Investment Thesis / Scorecard chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_breakdown(
    dimension_names: list[str],
    scores: list[int | None],
    details: list[str],
) -> go.Figure:
    """
    Horizontal bar chart showing the score (1–5) per dimension,
    colour-coded from red (1) to green (5).
    None scores are shown as grey "N/A".
    """
    from utils.scoring import SCORE_COLORS, SCORE_LABELS

    y      = []
    x      = []
    colors = []
    texts  = []

    for name, sc, det in zip(dimension_names, scores, details):
        y.append(name)
        if sc is None:
            x.append(0)
            colors.append("#4b5563")
            texts.append("  N/A — insufficient data")
        else:
            x.append(sc)
            colors.append(SCORE_COLORS.get(sc, "#94a3b8"))
            texts.append(f"  {SCORE_LABELS.get(sc, '')} — {det}")

    fig = go.Figure(go.Bar(
        y=y, x=x,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        text=texts,
        textposition="inside",
        insidetextanchor="start",
        hovertemplate="%{y}: %{x}/5<extra></extra>",
        width=0.55,
    ))

    # Reference line at 3 (neutral)
    fig.add_vline(x=3, line_dash="dot", line_color="rgba(150,150,150,0.4)", line_width=1)

    fig.update_layout(
        xaxis=dict(
            range=[0, 5.4],
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1 Poor", "2 Weak", "3 Neutral", "4 Good", "5 Excellent"],
            showgrid=True, gridcolor="rgba(150,150,150,0.15)",
        ),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(t=20, b=20, l=140, r=20),
        showlegend=False,
    )
    return fig
