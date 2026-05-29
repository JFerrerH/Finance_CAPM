import streamlit as st
import datetime
import numpy as np
import pandas as pd

from utils.data import download_ticker_data, get_stock_info
from utils.fmp import get_sector_pe, get_fmp_income_statement
from utils.capm import calculate_capm
from utils.performance import calculate_performance_metrics, calculate_rolling_beta, calculate_var_cvar
from utils.montecarlo import run_monte_carlo
from utils.markowitz import run_efficient_frontier, get_optimal_portfolios
from utils.fundamentals import parse_fundamentals, BENCHMARKS
from utils.valuation import (
    calculate_5yr_cagr_from_fmp,
    calculate_pe_fair_value,
    calculate_ddm,
    calculate_dcf,
    calculate_lynch_fair_value,
)
from utils.charts import (
    plot_sml,
    plot_monthly_returns,
    plot_regression,
    plot_valuation_comparison,
    plot_rolling_beta,
    plot_drawdown,
    plot_return_distribution,
    plot_var_distribution,
    plot_monte_carlo,
    plot_terminal_distribution,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_weights_pie,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_portfolio_returns(tickers_tuple, start, end):
    """Download and align monthly returns for a basket of tickers."""
    frames = {}
    for t in tickers_tuple:
        try:
            df = download_ticker_data(t, start, end, "1mo")
            frames[t] = df["Close"].pct_change().rename(t)
        except Exception:
            pass
    if len(frames) < 2:
        return None
    return pd.concat(frames.values(), axis=1).dropna()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAPM Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Layout */
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

/* Hide footer */
footer { visibility: hidden; }

/* Metric cards — use Streamlit theme variables so they adapt to dark/light */
div[data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}

/* Sidebar branding */
.sidebar-brand {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-color);
    letter-spacing: -0.5px;
}
.sidebar-sub {
    font-size: 0.78rem;
    color: var(--text-color);
    opacity: 0.55;
    margin-top: -4px;
    margin-bottom: 12px;
}

/* Section label above inputs */
.input-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-color);
    opacity: 0.45;
    margin-bottom: 4px;
}

/* Page header strip */
.page-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 1.2rem;
}
.page-header h2 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-color);
}
.page-header .ticker-badge {
    background: rgba(99, 102, 241, 0.12);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 2px 10px;
}

/* Stats card used in Analysis */
.stat-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-color);
    opacity: 0.55;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-color);
    margin-top: 2px;
}

/* Divider with label */
.section-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.5rem 0 1rem 0;
    color: var(--text-color);
    opacity: 0.45;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.section-divider::before, .section-divider::after {
    content: "";
    flex: 1;
    border-top: 1px solid rgba(148, 163, 184, 0.25);
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
INDEX_NAMES = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones"}

with st.sidebar:
    st.markdown('<p class="sidebar-brand">📈 CAPM Calculator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Capital Asset Pricing Model</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<p class="input-label">Configuration</p>', unsafe_allow_html=True)
    ticker_accion = st.text_input("Stock Ticker", "AAPL", placeholder="e.g. AAPL, MSFT")
    ticker_indice = st.selectbox(
        "Benchmark Index", ["^GSPC", "^IXIC", "^DJI"],
        format_func=lambda x: f"{INDEX_NAMES[x]} ({x})",
    )
    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
    bond_ticker = st.text_input("Risk-Free Rate", "^TNX", placeholder="e.g. ^TNX")

    st.divider()
    st.markdown('<p class="input-label">Navigate</p>', unsafe_allow_html=True)
    page = st.radio(
        "page",
        ["📊  Dashboard", "📉  Analysis", "📐  Performance", "🎲  Risk",
         "📦  Portfolio", "🏦  Fundamentals", "💰  Valuation", "🗃  Raw Data"],
        label_visibility="collapsed",
    )

# ── Data & calculations ───────────────────────────────────────────────────────
interval   = "1mo"
today_date = datetime.date.today()

data_accion = download_ticker_data(ticker_accion, start_date, today_date, interval)
data_indice = download_ticker_data(ticker_indice, start_date, today_date, interval)

data_accion["Monthly_Return_Stock"] = data_accion["Close"].pct_change()
data_indice["Monthly_Return_Index"] = data_indice["Close"].pct_change()
data = data_accion.join(data_indice["Monthly_Return_Index"]).dropna()

try:
    Rf_data = download_ticker_data(bond_ticker, start_date, today_date, interval)
    Rf = Rf_data["Close"].iloc[-1] / 100
except Exception as e:
    st.sidebar.error(f"Bond data unavailable: {e}")
    Rf = 0.04

capm  = calculate_capm(data, Rf)
beta  = capm["beta"]
Rm    = capm["Rm"]
CAPM  = capm["capm_return"]

perf             = calculate_performance_metrics(data, Rf, beta)
rolling_beta     = calculate_rolling_beta(data)
var_results      = calculate_var_cvar(data["Monthly_Return_Stock"].dropna())
current_price_mc = float(data_accion["Close"].iloc[-1])

# ── Helper ────────────────────────────────────────────────────────────────────
def page_header(title, ticker):
    st.markdown(
        f'<div class="page-header"><h2>{title}</h2>'
        f'<span class="ticker-badge">{ticker.upper()}</span></div>',
        unsafe_allow_html=True,
    )

def section_divider(label):
    st.markdown(f'<div class="section-divider">{label}</div>', unsafe_allow_html=True)

def stat_card(label, value):
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

def price_delta(fair, current):
    if fair and current:
        pct = (fair - current) / current * 100
        return f"{pct:+.1f}%"
    return None

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    page_header("Dashboard", ticker_accion)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAPM Expected Return", f"{CAPM:.2f}%")
    c2.metric("Beta (β)", f"{beta:.2f}",
              delta=f"{beta - 1:+.2f} vs market", delta_color="off")
    c3.metric("Risk-Free Rate", f"{Rf * 100:.2f}%")
    c4.metric("Market Return (Rm)", f"{Rm * 100:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sharpe Ratio",  f"{perf['sharpe']:.2f}"  if perf['sharpe']  is not None else "N/A",
              help="(Ann. Return − Rf) / Ann. Volatility")
    c6.metric("Sortino Ratio", f"{perf['sortino']:.2f}" if perf['sortino'] is not None else "N/A",
              help="(Ann. Return − Rf) / Downside Deviation")
    c7.metric("Max Drawdown",  f"{perf['max_drawdown']:.2%}")
    c8.metric("R² (Fit)",      f"{capm['r_squared']:.4f}",
              help="% of stock variance explained by the market")

    section_divider("Security Market Line")
    st.plotly_chart(plot_sml(Rf, Rm, beta, CAPM, ticker_accion), use_container_width=True, theme="streamlit")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Analysis
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📉  Analysis":
    page_header("Return Analysis", ticker_accion)

    section_divider("Monthly Returns")
    st.plotly_chart(plot_monthly_returns(data), use_container_width=True, theme="streamlit")

    section_divider("OLS Regression — Beta Estimation")
    chart_col, stats_col = st.columns([3, 1], gap="large")

    with chart_col:
        st.plotly_chart(
            plot_regression(capm["X"], capm["Y"], capm["y_pred"]),
            use_container_width=True,
            theme="streamlit",
        )

    with stats_col:
        st.markdown("<br>", unsafe_allow_html=True)
        stat_card("Beta (β)", f"{beta:.4f}")
        stat_card("Alpha (α)", f"{capm['intercept']:.4f}")
        stat_card("Correlation", f"{capm['correlation']:.4f}")
        stat_card("R²", f"{capm['r_squared']:.4f}")
        stat_card("Mkt. Return (Rm)", f"{Rm * 100:.2f}%")

    section_divider("OLS Diagnostic Tests")
    d1, d2, d3 = st.columns(3)
    d1.metric("α p-value",
              f"{capm['alpha_pvalue']:.4f}",
              delta="Significant" if capm['alpha_pvalue'] < 0.05 else "Not significant",
              delta_color="normal" if capm['alpha_pvalue'] < 0.05 else "off",
              help="Probability alpha ≠ 0 by chance. < 0.05 = statistically significant Jensen's Alpha.")
    d2.metric("β p-value",
              f"{capm['beta_pvalue']:.4f}",
              delta="Significant" if capm['beta_pvalue'] < 0.05 else "Not significant",
              delta_color="normal" if capm['beta_pvalue'] < 0.05 else "off",
              help="Probability beta ≠ 0 by chance. Should be < 0.05 for a valid CAPM estimate.")
    d3.metric("Jensen's α (Ann.)",
              f"{capm['jensen_alpha_annual']:.2%}",
              help="Annualised excess return vs. CAPM prediction. Positive = outperformance.")

    section_divider("Risk Decomposition")
    r1, r2, r3 = st.columns(3)
    r1.metric("Systematic Risk",   f"{capm['systematic_pct']:.1%}",
              help="Share of total variance explained by market movements (= R²).")
    r2.metric("Unsystematic Risk", f"{capm['unsystematic_pct']:.1%}",
              help="Firm-specific (idiosyncratic) risk that diversification can eliminate.")
    r3.metric("R² (Goodness of Fit)", f"{capm['r_squared']:.4f}")

    section_divider("Rolling 12-Month Beta")
    if len(rolling_beta) > 0:
        st.plotly_chart(plot_rolling_beta(rolling_beta, beta), use_container_width=True, theme="streamlit")
    else:
        st.info("Not enough data for rolling beta (need > 12 months).")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Performance & Risk
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📐  Performance":
    page_header("Performance & Risk", ticker_accion)

    section_divider("Risk-Adjusted Return Ratios")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Sharpe Ratio",
              f"{perf['sharpe']:.2f}" if perf['sharpe'] is not None else "N/A",
              help="(Ann. Return − Rf) / Ann. Volatility. > 1 = good, > 2 = excellent.")
    p2.metric("Treynor Ratio",
              f"{perf['treynor']:.2f}" if perf['treynor'] is not None else "N/A",
              help="(Ann. Return − Rf) / β. Reward per unit of systematic risk.")
    p3.metric("Sortino Ratio",
              f"{perf['sortino']:.2f}" if perf['sortino'] is not None else "N/A",
              help="(Ann. Return − Rf) / Downside Deviation. Penalises only negative volatility.")
    p4.metric("Calmar Ratio",
              f"{perf['calmar']:.2f}" if perf['calmar'] is not None else "N/A",
              help="Ann. Return / |Max Drawdown|. > 1 = strong risk-adjusted performance.")
    p5.metric("Jensen's α (Ann.)",
              f"{capm['jensen_alpha_annual']:.2%}",
              help="Annualised excess return above the CAPM prediction.")

    section_divider("Summary Stats")
    s1, s2, s3 = st.columns(3)
    s1.metric("Ann. Return",    f"{perf['ann_return']:.2%}")
    s2.metric("Ann. Volatility", f"{perf['ann_vol']:.2%}")
    s3.metric("Max Drawdown",   f"{perf['max_drawdown']:.2%}")

    section_divider("Underwater Chart")
    st.plotly_chart(plot_drawdown(perf["drawdown_series"]), use_container_width=True, theme="streamlit")

    section_divider("Return Distribution")
    st.plotly_chart(
        plot_return_distribution(data["Monthly_Return_Stock"].dropna()),
        use_container_width=True,
        theme="streamlit",
    )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Risk
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎲  Risk":
    page_header("Risk Analysis", ticker_accion)

    returns = data["Monthly_Return_Stock"].dropna()

    # ── VaR / CVaR ─────────────────────────────────────────────────────────────
    section_divider("Value at Risk — Historical Method")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric(
        "VaR 95% (Monthly)", f"{var_results['var_95']:.2%}",
        help="At 95% confidence, monthly loss will not exceed this threshold.",
    )
    v2.metric(
        "CVaR 95% (Monthly)", f"{var_results['cvar_95']:.2%}",
        help="Average loss in the worst 5% of months (Expected Shortfall).",
    )
    v3.metric(
        "VaR 99% (Monthly)", f"{var_results['var_99']:.2%}",
        help="At 99% confidence, monthly loss will not exceed this threshold.",
    )
    v4.metric(
        "CVaR 99% (Monthly)", f"{var_results['cvar_99']:.2%}",
        help="Average loss in the worst 1% of months.",
    )

    section_divider("Return Distribution with VaR Thresholds")
    st.plotly_chart(
        plot_var_distribution(returns, var_results),
        use_container_width=True, theme="streamlit",
    )

    # ── Monte Carlo ─────────────────────────────────────────────────────────────
    section_divider("Monte Carlo Simulation — Geometric Brownian Motion")
    mc1, mc2 = st.columns(2)
    with mc1:
        n_sims   = st.slider("Simulations", 100, 1000, 500, step=100)
    with mc2:
        n_months = st.slider("Horizon (months)", 6, 60, 24, step=6)

    _, pct_paths, final_prices = run_monte_carlo(
        returns, current_price_mc, n_sims, n_months,
    )
    future_dates = pd.date_range(
        start=data_accion.index[-1], periods=n_months + 1, freq="MS",
    )

    st.plotly_chart(
        plot_monte_carlo(pct_paths, future_dates, current_price_mc, ticker_accion),
        use_container_width=True, theme="streamlit",
    )

    # ── Terminal distribution ───────────────────────────────────────────────────
    section_divider("Terminal Price Distribution")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Current Price",   f"${current_price_mc:.2f}")
    t2.metric("10th Percentile", f"${np.percentile(final_prices, 10):.2f}",
              help="Pessimistic scenario: only 10% of paths finish below this.")
    t3.metric("Median (50th)",   f"${np.percentile(final_prices, 50):.2f}")
    t4.metric("90th Percentile", f"${np.percentile(final_prices, 90):.2f}",
              help="Optimistic scenario: only 10% of paths finish above this.")

    st.plotly_chart(
        plot_terminal_distribution(final_prices, current_price_mc),
        use_container_width=True, theme="streamlit",
    )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Portfolio Optimization
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📦  Portfolio":
    page_header("Portfolio Optimization", "Markowitz Efficient Frontier")

    # ── Inline ticker input (page-level config, not global) ─────────────────────
    with st.container():
        inp_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
        with inp_col:
            portfolio_input = st.text_input(
                "Tickers to analyse",
                value="AAPL, MSFT, GOOGL, AMZN",
                placeholder="e.g. AAPL, MSFT, TSLA, AMZN, NVDA",
                help="Enter 2–10 comma-separated ticker symbols.",
                label_visibility="visible",
            )
        with btn_col:
            run_btn = st.button("Analyse", type="primary", use_container_width=True)

    raw_tickers = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]

    if len(raw_tickers) < 2:
        st.info("Enter at least **2 tickers** above and click **Analyse**.")
        st.stop()

    badge = "  ·  ".join(raw_tickers[:4]) + ("  ·  …" if len(raw_tickers) > 4 else "")

    # ── Load data ───────────────────────────────────────────────────────────────
    with st.spinner(f"Loading data for {', '.join(raw_tickers)}…"):
        port_returns = load_portfolio_returns(tuple(raw_tickers), start_date, today_date)

    if port_returns is None or port_returns.shape[1] < 2:
        st.error("Could not load enough data. Check your ticker symbols and date range.")
        st.stop()

    tickers_used = port_returns.columns.tolist()
    failed = [t for t in raw_tickers if t not in tickers_used]
    if failed:
        st.warning(f"Could not load: {', '.join(failed)} — proceeding without them.")

    mean_returns = port_returns.mean()
    cov_matrix   = port_returns.cov()

    # Individual asset annualised stats
    asset_stats = []
    for t in tickers_used:
        ann_ret = float((1 + mean_returns[t]) ** 12 - 1)
        ann_vol = float(np.sqrt(cov_matrix.loc[t, t] * 12))
        sr      = (ann_ret - Rf) / ann_vol if ann_vol > 0 else 0.0
        asset_stats.append({"name": t, "return_ann": ann_ret, "vol_ann": ann_vol, "sharpe": sr})

    # ── Individual asset stats table ────────────────────────────────────────────
    section_divider("Individual Assets")
    asset_cols = st.columns(len(tickers_used))
    for col, a in zip(asset_cols, asset_stats):
        col.metric(
            a["name"],
            f"{a['return_ann']:.2%}",
            delta=f"Vol {a['vol_ann']:.2%}  |  SR {a['sharpe']:.2f}",
            delta_color="off",
        )

    # ── Correlation matrix ──────────────────────────────────────────────────────
    section_divider("Correlation Matrix")
    st.plotly_chart(
        plot_correlation_heatmap(port_returns),
        use_container_width=True, theme="streamlit",
    )

    # ── Efficient frontier ──────────────────────────────────────────────────────
    section_divider("Efficient Frontier")
    with st.spinner("Running Monte Carlo (5 000 portfolios)…"):
        frontier_df = run_efficient_frontier(mean_returns, cov_matrix, Rf)

    optimals   = get_optimal_portfolios(frontier_df)
    max_sharpe = optimals["max_sharpe"]
    min_vol    = optimals["min_vol"]

    st.caption("💡 Click any dot to inspect that portfolio’s composition.")
    frontier_event = st.plotly_chart(
        plot_efficient_frontier(frontier_df, max_sharpe, min_vol, Rf, asset_stats, tickers_used),
        use_container_width=True, theme="streamlit",
        on_select="rerun", key="frontier_chart",
    )

    # ── Click-to-inspect panel ─────────────────────────────────────────────────────────
    sel_points = (
        frontier_event.selection.points
        if frontier_event and hasattr(frontier_event, "selection")
        else []
    )
    if sel_points and sel_points[0].get("curve_number", -1) == 0:
        pt          = sel_points[0]
        point_idx   = pt["point_number"]              # stable: fixed seed in run_efficient_frontier
        sel_row     = frontier_df.iloc[point_idx]
        sel_weights = sel_row[tickers_used].values.astype(float)
        sel_ret     = float(sel_row["Return"])
        sel_vol     = float(sel_row["Volatility"])
        sel_sr      = float(sel_row["Sharpe"])

        section_divider("🔍 Selected Portfolio")
        pie_col, stats_col = st.columns([2, 3], gap="large")
        with pie_col:
            st.plotly_chart(
                plot_weights_pie(tickers_used, sel_weights),
                use_container_width=True, theme="streamlit",
            )
        with stats_col:
            st.markdown("**Portfolio metrics**")
            ma, mb, mc = st.columns(3)
            ma.metric("Expected Return", f"{sel_ret:.2%}")
            mb.metric("Volatility",      f"{sel_vol:.2%}")
            mc.metric("Sharpe Ratio",    f"{sel_sr:.2f}")
            st.markdown("**Weights**")
            weight_data = pd.DataFrame({
                "Ticker": tickers_used,
                "Weight": [f"{w:.2%}" for w in sel_weights],
            })
            st.dataframe(weight_data, use_container_width=True, hide_index=True)
    elif sel_points:
        # Clicked a special marker (CML, asset dot, optimal) — not a random portfolio
        pass  # no-op; frontier stats already shown below

    # ── Optimal portfolio stats ─────────────────────────────────────────────────
    section_divider("Optimal Portfolios")
    ms_col, mv_col = st.columns(2)

    with ms_col:
        st.markdown("**⭐ Maximum Sharpe Ratio**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Return", f"{float(max_sharpe['Return']):.2%}")
        m2.metric("Volatility",      f"{float(max_sharpe['Volatility']):.2%}")
        m3.metric("Sharpe Ratio",    f"{float(max_sharpe['Sharpe']):.2f}")

    with mv_col:
        st.markdown("**💎 Minimum Volatility**")
        v1, v2, v3 = st.columns(3)
        v1.metric("Expected Return", f"{float(min_vol['Return']):.2%}")
        v2.metric("Volatility",      f"{float(min_vol['Volatility']):.2%}")
        v3.metric("Sharpe Ratio",    f"{float(min_vol['Sharpe']):.2f}")

    # ── Weight comparison ───────────────────────────────────────────────────────
    section_divider("Portfolio Weights")
    ms_w = max_sharpe[tickers_used].values.astype(float)
    mv_w = min_vol[tickers_used].values.astype(float)
    st.plotly_chart(
        plot_portfolio_weights(tickers_used, ms_w, mv_w),
        use_container_width=True, theme="streamlit",
    )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Fundamentals
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🏦  Fundamentals":
    page_header("Fundamental Ratios", ticker_accion)

    info = get_stock_info(ticker_accion)
    if not info:
        st.warning("⚠️ Could not fetch stock info from Yahoo Finance (rate limit). Try again in a few minutes.")
        st.stop()

    fund = parse_fundamentals(info)

    # Helper to format a value with its unit
    def _disp(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "N/A"

    def _render_group(title, metrics_dict, suffix=""):
        section_divider(title)
        cols = st.columns(len(metrics_dict))
        for col, (label, value) in zip(cols, metrics_dict.items()):
            col.metric(label, _disp(value, suffix), help=BENCHMARKS.get(label))

    _render_group("Valuation Multiples", fund["valuation"], suffix="×")
    _render_group("Profitability (%)",   fund["profitability"], suffix="%")
    _render_group("Leverage & Liquidity", fund["leverage"], suffix="×")

    # Growth — mixed units (% and $), render individually
    section_divider("Growth & Earnings")
    g = fund["growth"]
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Revenue Growth (YoY)",  _disp(g["Revenue Growth (YoY)"], "%"),
              help=BENCHMARKS.get("Revenue Growth (YoY)"))
    g2.metric("Earnings Growth (YoY)", _disp(g["Earnings Growth (YoY)"], "%"),
              help=BENCHMARKS.get("Earnings Growth (YoY)"))
    g3.metric("EPS (TTM)",     f"${g['EPS (TTM)']:.2f}" if g["EPS (TTM)"] is not None else "N/A",
              help=BENCHMARKS.get("EPS (TTM)"))
    g4.metric("Forward EPS",   f"${g['Forward EPS']:.2f}" if g["Forward EPS"] is not None else "N/A",
              help=BENCHMARKS.get("Forward EPS"))

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Valuation
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💰  Valuation":
    page_header("Fair Value Estimations", ticker_accion)

    info = get_stock_info(ticker_accion)
    if not info:
        st.warning("⚠️ Could not fetch stock info from Yahoo Finance (rate limit). Try again in a few minutes.")
        st.stop()

    current_price    = info.get("currentPrice")
    eps              = info.get("trailingEps")
    dividend         = info.get("dividendRate", 0) or 0
    required_return  = CAPM / 100 if CAPM else 0.10
    fcf              = info.get("freeCashflow", 0)
    shares_out       = info.get("sharesOutstanding", 1)

    api_key   = st.secrets["fmp"]["api_key"]
    sector_pe = get_sector_pe(ticker_accion, api_key) or 25
    if sector_pe <= 0:
        sector_pe = 25

    pe_fair_price                          = calculate_pe_fair_value(eps, sector_pe)
    ddm_price                              = calculate_ddm(dividend, required_return)
    dcf_price, annual_fcf, total_val, dcf_warn = calculate_dcf(fcf, shares_out, required_return)

    fmp_df = get_fmp_income_statement(ticker_accion, api_key)
    if fmp_df is not None and eps:
        gr = calculate_5yr_cagr_from_fmp(fmp_df, "ebitda")
        if gr is not None:
            lynch_price, lynch_caption, lynch_warn = calculate_lynch_fair_value(eps, gr)
        else:
            lynch_price, lynch_caption, lynch_warn = None, None, "⚠️ Not enough EBITDA data for Peter Lynch."
    else:
        lynch_price, lynch_caption, lynch_warn = None, None, "⚠️ Peter Lynch: missing EPS or FMP data."

    # Warnings
    for msg in filter(None, [dcf_warn, lynch_warn]):
        st.warning(msg)

    # Comparison chart
    section_divider("Fair Value vs. Current Price")
    st.plotly_chart(
        plot_valuation_comparison(current_price, pe_fair_price, ddm_price, dcf_price, lynch_price),
        use_container_width=True,
        theme="streamlit",
    )

    # Metric cards
    section_divider("Method Breakdown")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price",
              f"${current_price:.2f}" if current_price else "N/A")
    c2.metric("P/E Method",
              f"${pe_fair_price:.2f}" if pe_fair_price else "N/A",
              delta=price_delta(pe_fair_price, current_price))
    c3.metric("DCF Method",
              f"${dcf_price:.2f}" if dcf_price else "N/A",
              delta=price_delta(dcf_price, current_price))
    c4.metric("DDM Method",
              f"${ddm_price:.2f}" if ddm_price else "N/A",
              delta=price_delta(ddm_price, current_price))
    c5.metric("Peter Lynch",
              f"${lynch_price:.2f}" if lynch_price else "N/A",
              delta=price_delta(lynch_price, current_price),
              help=lynch_caption or "")

    # Expanders
    if dcf_price:
        with st.expander("📋 DCF Detail"):
            dc1, dc2 = st.columns(2)
            dc1.metric("Annual FCF", f"${annual_fcf:,.0f}")
            dc2.metric("Total Firm Value", f"${total_val:,.0f}")

    with st.expander("🔍 Raw Inputs"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**EPS:**", eps)
            st.write("**Sector P/E used:**", sector_pe)
            st.write("**Dividend Rate:**", dividend)
        with col_b:
            st.write("**Required Return (CAPM):**", f"{required_return:.2%}")
            st.write("**Free Cash Flow:**", f"${fcf:,.0f}" if fcf else "N/A")
            st.write("**Shares Outstanding:**", f"{shares_out:,.0f}")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Raw Data
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🗃  Raw Data":
    page_header("Raw Data", ticker_accion)

    tab_stock, tab_index = st.tabs([
        f"📈 {ticker_accion.upper()} — Stock",
        f"📊 {INDEX_NAMES.get(ticker_indice, ticker_indice)} — Index",
    ])
    with tab_stock:
        st.dataframe(
            data_accion[["Close", "Monthly_Return_Stock"]].dropna(),
            use_container_width=True,
        )
    with tab_index:
        st.dataframe(
            data_indice[["Close", "Monthly_Return_Index"]].dropna(),
            use_container_width=True,
        )

