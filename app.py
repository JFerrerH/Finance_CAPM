import streamlit as st
import datetime
import numpy as np
import pandas as pd

from utils.data import download_ticker_data, get_stock_info
from utils.fmp import get_sector_pe, get_fmp_income_statement
from utils.capm import calculate_capm
from utils.performance import (
    calculate_performance_metrics, calculate_rolling_beta,
    calculate_var_cvar, calculate_momentum_metrics,
)
from utils.montecarlo import run_monte_carlo
from utils.markowitz import run_efficient_frontier, get_optimal_portfolios
from utils.fundamentals import parse_fundamentals, BENCHMARKS
from utils.financials import (
    get_financial_statements,
    extract_income_data, extract_cashflow_data, extract_balance_data,
    yoy_pct, fmt_large, calc_cagr,
)
from utils.valuation import (
    calculate_5yr_cagr_from_fmp,
    calculate_pe_fair_value,
    calculate_ddm,
    calculate_dcf,
    calculate_lynch_fair_value,
)
from utils.macro import get_macro_data, compute_cycle_signals, macro_implication
from utils.fred import get_economic_data
from utils.scoring import (
    score_alpha, score_risk_return, score_momentum,
    score_financial_health, score_valuation, score_macro_fit,
    get_verdict, build_bull_bear,
    SCORE_COLORS, SCORE_LABELS,
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
    plot_revenue_earnings_trend,
    plot_sankey,
    plot_cashflow_bars,
    plot_capital_structure,
    plot_dalio_quadrant,
    plot_macro_history,
    plot_score_breakdown,
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
        ["📊  Dashboard", "📍  Thesis", "📉  Analysis", "📐  Performance", "🎲  Risk",
         "🌍  Macro", "📦  Portfolio", "🏦  Fundamentals", "🏥  Financial Health",
         "💰  Valuation", "🗃  Raw Data"],
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
mom_metrics      = calculate_momentum_metrics(data)
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

    # ── Liquidity / market-cap guardrail ─────────────────────────────────────
    _info_dash = get_stock_info(ticker_accion)
    if _info_dash:
        _mcap   = _info_dash.get("marketCap")
        _avgvol = _info_dash.get("averageVolume")
        if _mcap and _mcap < 300_000_000:
            st.warning(
                f"**Micro-cap alert:** market cap ${_mcap/1e6:.0f}M. "
                "CAPM assumptions (continuous liquidity, no price impact) may not hold. "
                "Beta and alpha estimates carry wider uncertainty for thinly traded securities.",
                icon="⚠️",
            )
        elif _mcap and _mcap < 1_000_000_000:
            st.info(
                f"**Small-cap:** market cap ${_mcap/1e6:.0f}M. "
                "Estimates are valid but liquidity risk and estimation noise are higher than large-caps.",
                icon="ℹ️",
            )
        if _avgvol and _avgvol < 500_000:
            st.warning(
                f"**Low liquidity:** avg. daily volume {_avgvol:,} shares. "
                "Bid-ask spreads may inflate measured volatility and distort monthly return data.",
                icon="⚠️",
            )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAPM Expected Return", f"{CAPM:.2f}%")
    _ci_low  = capm.get("beta_ci_low",  beta)
    _ci_high = capm.get("beta_ci_high", beta)
    c2.metric("Beta (β)", f"{beta:.2f}",
              delta=f"95% CI  [{_ci_low:.2f}, {_ci_high:.2f}]", delta_color="off",
              help="OLS point estimate ± 1.96 × standard error. Wide interval = unreliable beta.")
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
# PAGE: Investment Thesis
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📍  Thesis":
    page_header("Investment Thesis", ticker_accion)

    # ── Load additional data needed for scoring ───────────────────────────────
    info    = get_stock_info(ticker_accion)
    stmts   = get_financial_statements(ticker_accion)
    inc     = extract_income_data(stmts.get("income"))
    cf      = extract_cashflow_data(stmts.get("cashflow"))
    bal     = extract_balance_data(stmts.get("balance"))

    # Valuation inputs (mirror the Valuation page approach, all cached)
    current_price    = info.get("currentPrice")    if info else None
    eps              = info.get("trailingEps")     if info else None
    fcf_raw          = (info.get("freeCashflow") or 0) if info else 0
    shares_out       = (info.get("sharesOutstanding") or 1) if info else 1
    required_return  = CAPM / 100 if CAPM else 0.10

    api_key   = st.secrets.get("fmp", {}).get("api_key", "")
    sector_pe = get_sector_pe(ticker_accion, api_key) or 25
    if sector_pe <= 0:
        sector_pe = 25

    # Revenue CAGR drives DCF short-term growth (replaces the FCF-size tiers)
    rev_cagr  = calc_cagr(inc.get("revenue"))
    dividend  = (info.get("dividendRate") or 0) if info else 0

    pe_fair   = calculate_pe_fair_value(eps, sector_pe)
    dcf_tuple = calculate_dcf(fcf_raw, shares_out, required_return, rev_cagr)
    dcf_price = dcf_tuple[0] if dcf_tuple else None
    ddm_price = calculate_ddm(dividend, required_return, current_price)
    estimates = [p for p in [dcf_price, pe_fair, ddm_price] if p and p > 0]
    valuation_upside = (
        (sum(estimates) / len(estimates) - current_price) / current_price
        if estimates and current_price else None
    )

    # Macro cycle for context
    today_str   = datetime.date.today().isoformat()
    start_macro = (datetime.date.today() - datetime.timedelta(days=730)).isoformat()
    macro_data  = get_macro_data(start_macro, today_str)
    fred_api_key = st.secrets.get("fred", {}).get("api_key", "")
    fred_data   = get_economic_data(fred_api_key, start_macro, today_str) if fred_api_key else {}
    signals     = compute_cycle_signals(macro_data, fred_data)
    cycle_phase = signals.get("cycle_phase", "Transition")

    # ── Compute scores ────────────────────────────────────────────────────────
    sector    = (info.get("sector") or "") if info else ""
    s_alpha   = score_alpha(capm)
    s_risk    = score_risk_return(perf)
    s_mom     = score_momentum(mom_metrics)
    s_fin     = score_financial_health(inc, cf, bal, sector)
    s_val     = score_valuation(current_price, dcf_price, pe_fair, ddm_price)
    s_macro   = score_macro_fit(signals, beta, sector)

    score_dict = {
        "Alpha Quality":    s_alpha[0],
        "Risk / Return":    s_risk[0],
        "Momentum":         s_mom[0],
        "Financial Health": s_fin[0],
        "Valuation":        s_val[0],
        "Macro Fit":        s_macro[0],
    }
    verdict, verdict_color, avg_score = get_verdict(score_dict)

    # Bull / bear case
    bull_pts, bear_pts = build_bull_bear(
        capm, perf, var_results or {},
        inc, cf, bal,
        valuation_upside, cycle_phase, beta, sector,
    )

    # ── Verdict banner ────────────────────────────────────────────────────────
    geo_tag = " ⚠️" if signals.get("geo_adjustment_applied") else ""
    st.markdown(f"""
    <div style="
        background: {verdict_color}22;
        border: 2px solid {verdict_color};
        border-radius: 12px;
        padding: 28px 36px;
        text-align: center;
        margin-bottom: 24px;
    ">
        <div style="font-size: 0.9rem; opacity: 0.7; text-transform: uppercase;
                    letter-spacing: 2px; margin-bottom: 6px;">Investment Verdict</div>
        <div style="font-size: 2.8rem; font-weight: 800; color: {verdict_color};
                    letter-spacing: 1px;">{verdict}</div>
        <div style="font-size: 1rem; margin-top: 10px; opacity: 0.65;">
            Composite score: <b>{avg_score:.1f} / 5</b> &nbsp;·&nbsp; Macro: {signals.get('cycle_icon','')} {cycle_phase}{geo_tag}
        </div>
    </div>
    """, unsafe_allow_html=True)
    if signals.get("geopolitical_warning"):
        raw = signals.get("inflation_signal_raw", "Elevated")
        if signals.get("geo_adjustment_applied"):
            st.warning(
                f"**Geopolitical filter active:** The inflation signal was elevated by what appears to be "
                f"a supply-driven commodity shock (raw signal: {raw}). The cycle classification and macro score "
                "have been adjusted to reduce false-positive risk. See the Macro Environment page for full detail.",
                icon="⚠️",
            )
        else:
            st.info(
                "**Geopolitical overlay detected:** Market signals show a possible supply-driven commodity "
                "spike, but the inflation trend is persistent enough that no classification adjustment was made. "
                "The macro score reflects current cycle positioning including this geopolitical premium.",
                icon="ℹ️",
            )

    # ── Score breakdown chart ─────────────────────────────────────────────────
    section_divider("Score Breakdown")
    st.plotly_chart(
        plot_score_breakdown(
            list(score_dict.keys()),
            list(score_dict.values()),
            [s_alpha[1], s_risk[1], s_mom[1], s_fin[1], s_val[1], s_macro[1]],
        ),
        use_container_width=True,
        theme="streamlit",
    )

    # ── Bull / Bear case ──────────────────────────────────────────────────────
    section_divider("Bull Case vs Bear Case")
    col_bull, col_bear = st.columns(2)

    with col_bull:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);
             border-radius:8px;padding:16px 20px;">
        <div style="color:#10b981;font-weight:700;font-size:1.05rem;margin-bottom:12px;">
        ✅ Bull Case</div>
        """, unsafe_allow_html=True)
        if bull_pts:
            for pt in bull_pts:
                st.markdown(f"- {pt}")
        else:
            st.markdown("- No strong positive signals detected")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bear:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);
             border-radius:8px;padding:16px 20px;">
        <div style="color:#ef4444;font-weight:700;font-size:1.05rem;margin-bottom:12px;">
        ❌ Bear Case</div>
        """, unsafe_allow_html=True)
        if bear_pts:
            for pt in bear_pts:
                st.markdown(f"- {pt}")
        else:
            st.markdown("- No major risks identified")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Key metrics summary table ─────────────────────────────────────────────
    section_divider("Key Metrics at a Glance")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.metric("Annual Return",   f"{perf['ann_return']:.2%}")
        st.metric("Sharpe Ratio",    f"{perf['sharpe']:.2f}")
        st.metric("Beta",            f"{beta:.2f}")
    with kpi_col2:
        st.metric("Max Drawdown",    f"{perf['max_drawdown']:.2%}")
        st.metric("Jensen Alpha",    f"{capm.get('jensen_alpha_annual',0):.2%}/yr")
        st.metric("Macro Cycle",     f"{signals.get('cycle_icon','')} {cycle_phase}")
    with kpi_col3:
        st.metric("Current Price",   f"${current_price:,.2f}" if current_price else "—")
        if estimates:
            st.metric("Blended Fair Value", f"${sum(estimates)/len(estimates):,.2f}")
        if valuation_upside is not None:
            delta_str = f"{valuation_upside:+.1%} vs current"
            st.metric("Upside / Downside", delta_str)

    st.markdown(
        "<div style='opacity:0.45;font-size:0.78rem;margin-top:12px;'>"
        "This verdict is model-based and purely informational. Not financial advice.</div>",
        unsafe_allow_html=True,
    )

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
        stat_card("β 95% CI", f"[{capm.get('beta_ci_low', beta):.3f}, {capm.get('beta_ci_high', beta):.3f}]")
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

    section_divider("Rolling Beta")
    _window = st.slider(
        "Rolling window (months)", min_value=6, max_value=36, value=12, step=3,
        help="Shorter windows react faster but produce noisier estimates. 12–24M is standard.",
    )
    _rolling = calculate_rolling_beta(data, window=_window)
    if len(_rolling) > 0:
        st.plotly_chart(plot_rolling_beta(_rolling, beta), use_container_width=True, theme="streamlit")
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
    section_divider("Value at Risk — Historical vs Parametric (Monthly)")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric(
        "VaR 95% — Historical", f"{var_results['var_95']:.2%}",
        help="5th percentile of actual monthly returns. No distributional assumption.",
    )
    v2.metric(
        "VaR 95% — Parametric", f"{var_results.get('param_var_95', 0):.2%}",
        help="μ − 1.645σ assuming normality. Gap vs historical = fat-tail risk.",
    )
    v3.metric(
        "CVaR 95% (Monthly)", f"{var_results['cvar_95']:.2%}",
        help="Average loss in the worst 5% of months (Expected Shortfall).",
    )
    v4.metric(
        "VaR 99% — Historical", f"{var_results['var_99']:.2%}",
        help="1st percentile of actual monthly returns.",
    )

    _hist_95  = var_results.get("var_95", 0)
    _param_95 = var_results.get("param_var_95", 0)
    _gap      = abs(_hist_95 - _param_95)
    if _gap > 0.02:
        st.caption(
            f"⚠️ Historical VaR ({_hist_95:.1%}) differs from parametric ({_param_95:.1%}) "
            f"by {_gap:.1%} — the return distribution has meaningful fat tails. "
            "Parametric VaR underestimates true downside risk for this stock."
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

    _mc_signals = compute_cycle_signals(
        get_macro_data(
            (datetime.date.today() - datetime.timedelta(days=730)).isoformat(),
            datetime.date.today().isoformat(),
        )
    )
    _, pct_paths, final_prices = run_monte_carlo(
        returns, current_price_mc, n_sims, n_months,
        cycle_signals=_mc_signals, beta=beta,
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
# PAGE: Macro
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🌍  Macro":
    page_header("Macro Environment", "Ray Dalio Framework")

    today_str   = datetime.date.today().isoformat()
    start_macro = (datetime.date.today() - datetime.timedelta(days=1095)).isoformat()  # 3 years

    with st.spinner("Loading macro data…"):
        macro_data   = get_macro_data(start_macro, today_str)
        fred_api_key = st.secrets.get("fred", {}).get("api_key", "")
        fred_data    = get_economic_data(fred_api_key, start_macro, today_str) if fred_api_key else {}

    if not macro_data:
        st.warning("Could not load macro data. Please try again later.")
        st.stop()

    signals = compute_cycle_signals(macro_data, fred_data)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    section_divider("Current Market Conditions")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    rate = signals.get("rate_current")
    k1.metric("10Y Yield",
              f"{rate:.2f}%" if rate else "—",
              signals.get("rate_direction", "—"))

    vix = signals.get("vix_current")
    k2.metric("VIX",
              f"{vix:.1f}" if vix else "—",
              signals.get("vix_label", "—"))

    sp6 = signals.get("sp_6m")
    k3.metric("S&P 500 (6M)",
              f"{sp6:+.1f}%" if sp6 is not None else "—",
              signals.get("growth_signal", "—"))

    gold6 = signals.get("gold_6m")
    k4.metric("Gold (6M)",
              f"{gold6:+.1f}%" if gold6 is not None else "—",
              signals.get("inflation_signal", "—"))

    dxy = signals.get("dxy_3m")
    k5.metric("Dollar (3M)",
              f"{dxy:+.1f}%" if dxy is not None else "—",
              signals.get("dxy_direction", "—"))

    oil3 = signals.get("oil_3m")
    k6.metric("Oil (3M)",
              f"{oil3:+.1f}%" if oil3 is not None else "—")

    # ── Geopolitical supply-shock warning ────────────────────────────────────
    if signals.get("geopolitical_warning"):
        adj     = signals.get("geo_adjustment_applied", False)
        detail  = signals.get("geo_shock_detail", "")
        raw_sig = signals.get("inflation_signal_raw", "Elevated")
        if adj:
            adj_note = (
                f"The raw inflation signal was <b>{raw_sig}</b>. "
                "Based on the supply-shock pattern (oil/gold divergence + dollar strength + "
                "spike recency), the cycle classification has been adjusted to <b>Moderate</b> "
                "inflation to reduce false-positive risk. Monitor whether these signals persist — "
                "if they do, the original classification will reassert itself."
            )
            banner_color = "#f59e0b"
        else:
            adj_note = (
                "Geopolitical signals are present but the inflation trend appears persistent enough "
                "(built over 6+ months) that no classification adjustment was made. "
                "The current cycle phase reflects both the geopolitical overlay and the underlying macro. "
                "Watch for commodity prices normalising, which would confirm the shock is temporary."
            )
            banner_color = "#f97316"
        st.markdown(f"""
        <div style="
            background: rgba(245,158,11,0.08);
            border-left: 4px solid {banner_color};
            border-radius: 8px;
            padding: 16px 22px;
            margin: 16px 0 4px 0;
        ">
            <div style="font-size:0.82rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:1.2px;color:{banner_color};margin-bottom:8px;">
                ⚠️ Geopolitical Supply-Shock Signal Detected
            </div>
            <div style="font-size:0.9rem;line-height:1.65;opacity:0.88;margin-bottom:10px;">
                {detail}
            </div>
            <div style="font-size:0.85rem;line-height:1.6;opacity:0.68;font-style:italic;">
                {adj_note}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Dalio quadrant + cycle description ────────────────────────────────────
    section_divider("Economic Cycle Positioning")
    q_col, desc_col = st.columns([1, 1], gap="large")

    with q_col:
        gs  = signals.get("growth_score", 0.0)
        ins = signals.get("inflation_score", 0.0)
        st.plotly_chart(
            plot_dalio_quadrant(gs, ins, signals["cycle_phase"], signals["cycle_color"]),
            use_container_width=True, theme="streamlit",
        )

    with desc_col:
        phase_color = signals["cycle_color"]
        st.markdown(f"""
        <div style="
            background:{phase_color}18;
            border-left: 4px solid {phase_color};
            border-radius: 8px;
            padding: 20px 24px;
            margin-top: 60px;
        ">
            <div style="font-size:1.5rem;font-weight:700;color:{phase_color};margin-bottom:8px;">
                {signals.get('cycle_icon','')} {signals['cycle_phase']}
            </div>
            <div style="line-height:1.7;margin-bottom:16px;">{signals['cycle_desc']}</div>
            <div style="opacity:0.7;font-size:0.9rem;">
                <b>Equity positioning:</b> {signals.get('cycle_stocks','—')}<br>
                <b>Bond positioning:</b> {signals.get('cycle_bonds','—')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Cycle maturity ────────────────────────────────────────────────────────
    section_divider("Where Are We Within This Phase?")

    stage       = signals.get("cycle_stage", "Unclear")
    stage_pct   = signals.get("cycle_stage_pct", 50)
    stage_color = signals.get("cycle_stage_color", "#94a3b8")
    next_phase  = signals.get("cycle_next_phase", "—")
    next_icon   = signals.get("cycle_next_icon", "")
    watch       = signals.get("cycle_watch_signal", "—")
    t_risk      = signals.get("cycle_transition_risk", "—")
    stage_desc  = signals.get("cycle_stage_desc", "")

    m_left, m_right = st.columns([1, 1], gap="large")

    with m_left:
        # Progress bar + stage badge
        bar_filled = "█" * int(stage_pct / 10)
        bar_empty  = "░" * (10 - int(stage_pct / 10))
        st.markdown(f"""
        <div style="background:rgba(148,163,184,0.07);border:1px solid rgba(148,163,184,0.18);
             border-radius:10px;padding:20px 24px;">
            <div style="font-size:0.8rem;opacity:0.55;text-transform:uppercase;
                        letter-spacing:1.5px;margin-bottom:8px;">Cycle Stage</div>
            <div style="font-size:2rem;font-weight:800;color:{stage_color};margin-bottom:10px;">
                {stage}
            </div>
            <div style="font-family:monospace;font-size:1.1rem;color:{stage_color};margin-bottom:6px;">
                {bar_filled}{bar_empty} ~{stage_pct}%
            </div>
            <div style="font-size:0.82rem;opacity:0.55;">approximate progress through {signals.get('cycle_phase','—')}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_right:
        # Next phase + watch signal
        st.markdown(f"""
        <div style="background:rgba(148,163,184,0.07);border:1px solid rgba(148,163,184,0.18);
             border-radius:10px;padding:20px 24px;">
            <div style="font-size:0.8rem;opacity:0.55;text-transform:uppercase;
                        letter-spacing:1.5px;margin-bottom:8px;">Likely Next Phase</div>
            <div style="font-size:1.5rem;font-weight:700;margin-bottom:12px;">
                {next_icon} {next_phase}
            </div>
            <div style="font-size:0.88rem;opacity:0.75;margin-bottom:8px;">
                <b>Transition risk:</b> {t_risk}
            </div>
            <div style="font-size:0.88rem;opacity:0.75;">
                <b>📡 Watch:</b> {watch}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Maturity narrative
    st.markdown(f"""
    <div style="margin-top:16px;padding:16px 20px;
         background:rgba(148,163,184,0.05);
         border-left:3px solid {stage_color};border-radius:0 8px 8px 0;
         font-size:0.95rem;line-height:1.7;">
        {stage_desc}
    </div>
    """, unsafe_allow_html=True)

    # ── Economic Fundamentals (FRED) ─────────────────────────────────────────
    section_divider("Economic Fundamentals")
    if not signals.get("fred_available"):
        st.info(
            "**Connect FRED for real economic data** — CPI headline vs core, TIPS breakevens, "
            "unemployment, and industrial production. These are the signals that distinguish a "
            "geopolitical supply shock from structural inflation.\n\n"
            "Get a free API key at **fred.stlouisfed.org** and add it to "
            "`.streamlit/secrets.toml` under `[fred]` → `api_key`.",
            icon="🏦",
        )
    else:
        ef1, ef2, ef3, ef4, ef5, ef6 = st.columns(6)

        cpi_hl  = signals.get("cpi_headline_yoy")
        cpi_c   = signals.get("cpi_core_yoy")
        spread  = signals.get("cpi_spread")
        ef1.metric(
            "CPI Headline (YoY)",
            f"{cpi_hl:.1f}%" if cpi_hl is not None else "—",
            f"Spread vs core: {spread:+.1f}pp" if spread is not None else None,
        )
        ef2.metric(
            "CPI Core (YoY)",
            f"{cpi_c:.1f}%" if cpi_c is not None else "—",
            "ex food & energy",
        )

        unemp     = signals.get("unemployment")
        unemp_chg = signals.get("unemployment_3m_chg")
        ef3.metric(
            "Unemployment",
            f"{unemp:.1f}%" if unemp is not None else "—",
            f"{unemp_chg:+.2f}pp (3M)" if unemp_chg is not None else None,
        )

        tips5     = signals.get("tips_5y")
        tips5_chg = signals.get("tips_5y_3m_chg")
        ef4.metric(
            "TIPS 5Y Breakeven",
            f"{tips5:.2f}%" if tips5 is not None else "—",
            f"{tips5_chg * 100:+.0f} bp (3M)" if tips5_chg is not None else None,
        )

        tips10     = signals.get("tips_10y")
        tips10_chg = signals.get("tips_10y_3m_chg")
        ef5.metric(
            "TIPS 10Y Breakeven",
            f"{tips10:.2f}%" if tips10 is not None else "—",
            f"{tips10_chg * 100:+.0f} bp (3M)" if tips10_chg is not None else None,
        )

        indpro = signals.get("indpro_3m")
        ef6.metric(
            "Industrial Prod. (3M)",
            f"{indpro:+.1f}%" if indpro is not None else "—",
            "manufacturing activity proxy",
        )

        # CPI spread interpretation — the key supply-shock vs structural discriminator
        if spread is not None and cpi_hl is not None and cpi_c is not None:
            if spread > 1.5:
                st.caption(
                    f"⚠️ Headline CPI is running **{spread:.1f}pp above core** "
                    f"({cpi_hl:.1f}% vs {cpi_c:.1f}%) — inflation is concentrated in energy and food. "
                    "This pattern is typical of a supply-side shock, not structural demand-pull inflation. "
                    "If oil prices normalise, headline inflation should fall rapidly toward core."
                )
            elif spread < 0.5 and cpi_hl > 2.5:
                st.caption(
                    f"📌 Headline and core CPI are moving together ({spread:.1f}pp spread) — "
                    "broad-based inflation across goods and services. "
                    "This is a structural demand-pull signal, not a commodity spike."
                )

        # TIPS breakeven interpretation
        if tips5 is not None and tips5_chg is not None:
            tips5_bp = tips5_chg * 100
            if abs(tips5_bp) < 15 and signals.get("geopolitical_warning"):
                st.caption(
                    f"📌 5Y TIPS breakeven moved only {tips5_bp:+.0f}bp in 3 months despite the commodity "
                    "surge — bond markets are not pricing persistent inflation. "
                    "This supports the supply-shock (transient) interpretation."
                )
            elif tips5_bp > 30:
                st.caption(
                    f"⚠️ 5Y TIPS breakeven repriced +{tips5_bp:.0f}bp in 3 months — "
                    "inflation expectations are moving structurally, not just reacting to a spot price spike."
                )

    # ── Normalised trend chart ────────────────────────────────────────────────
    section_divider("Macro Trend History (3 Years)")
    available = [n for n in ["S&P 500", "Gold", "10Y Yield", "Oil", "Dollar"] if n in macro_data]
    if available:
        selected = st.multiselect(
            "Select indicators to compare",
            options=available,
            default=available[:3],
            key="macro_trend_select",
        )
        if selected:
            st.plotly_chart(
                plot_macro_history(macro_data, selected),
                use_container_width=True, theme="streamlit",
            )

    # ── Stock implication ─────────────────────────────────────────────────────
    section_divider(f"What This Means for {ticker_accion}")
    implication = macro_implication(signals["cycle_phase"], beta, ticker_accion)
    st.markdown(f"""
    <div style="
        background: rgba(148,163,184,0.07);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 8px;
        padding: 20px 24px;
        line-height: 1.8;
        font-size: 1.02rem;
    ">{implication}</div>
    """, unsafe_allow_html=True)

    # ── Ray Dalio reference ───────────────────────────────────────────────────
    with st.expander("About the Ray Dalio Framework"):
        st.markdown("""
        **Ray Dalio's Economic Machine** identifies two key debt cycles that drive asset prices:

        - **Short-term debt cycle** (~5-8 years): driven by credit expansion/contraction. Central banks
          control this via interest rate policy.
        - **Long-term debt cycle** (~75-100 years): driven by the cumulative build-up of debt.
          Ends in deleveraging (deflationary or inflationary).

        The **four macro environments** each favour different asset classes:

        | Environment | Growth | Inflation | Favoured Assets |
        |---|---|---|---|
        | 🌤 Goldilocks | Rising | Falling/Low | Stocks, High-beta equities |
        | 🔥 Inflationary Boom | Rising | Rising | Commodities, Energy, Banks |
        | ⚠️ Stagflation | Falling | Rising | Gold, Commodities, Short-duration |
        | ❄️ Deflationary Bust | Falling | Falling | Long bonds, Gold, Defensive equities |

        *Note: This page uses market-observable proxies (equity returns, gold, rates, VIX) to estimate
        the current regime. It is not a real-time macro model and should be one input among many.*
        """)

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
# PAGE: Financial Health
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🏥  Financial Health":
    page_header("Financial Health", ticker_accion)

    with st.spinner("Loading financial statements…"):
        stmts = get_financial_statements(ticker_accion)

    income_df  = stmts.get("income",   None)
    balance_df = stmts.get("balance",  None)
    cf_df      = stmts.get("cashflow", None)

    if (income_df is None or income_df.empty) and \
       (balance_df is None or balance_df.empty):
        st.warning("⚠️ Could not load financial statements from Yahoo Finance. Try again later.")
        st.stop()

    inc  = extract_income_data(income_df)
    cf   = extract_cashflow_data(cf_df)
    bal  = extract_balance_data(balance_df)

    # ── KPI strip ───────────────────────────────────────────────────────────────
    section_divider("Key Figures (Most Recent Year)")
    k1, k2, k3, k4, k5 = st.columns(5)

    rev_yoy  = yoy_pct(inc.get("revenue"))
    ni_yoy   = yoy_pct(inc.get("net_income"))
    fcf_val  = cf.get("fcf")
    cash_val = bal.get("cash")
    debt_val = bal.get("total_debt")

    k1.metric("Revenue",
              fmt_large(inc["revenue"][-1] if inc.get("revenue") else None),
              delta=f"{rev_yoy:.1%} YoY" if rev_yoy is not None else None)
    k2.metric("Net Income",
              fmt_large(inc["net_income"][-1] if inc.get("net_income") else None),
              delta=f"{ni_yoy:.1%} YoY" if ni_yoy is not None else None)
    k3.metric("Free Cash Flow",
              fmt_large(fcf_val[-1] if fcf_val else None))
    k4.metric("Cash & Equivalents",
              fmt_large(cash_val[-1] if cash_val else None))

    # Net Debt = Total Debt - Cash
    if debt_val and cash_val:
        net_debt = debt_val[-1] - cash_val[-1]
        k5.metric("Net Debt",
                  fmt_large(net_debt),
                  delta="Net Cash" if net_debt < 0 else "Net Debt",
                  delta_color="normal" if net_debt < 0 else "inverse")
    else:
        k5.metric("Total Debt", fmt_large(debt_val[-1] if debt_val else None))

    # ── Revenue & Earnings trend ─────────────────────────────────────────────
    section_divider("Revenue & Earnings")
    if inc.get("revenue"):
        st.plotly_chart(
            plot_revenue_earnings_trend(inc),
            use_container_width=True, theme="streamlit",
        )
    else:
        st.info("Revenue data not available.")

    # ── P&L Sankey ──────────────────────────────────────────────────────────
    section_divider("Where Does the Money Go? — P&L Flow")
    if inc.get("revenue") and inc.get("net_income"):
        st.plotly_chart(
            plot_sankey(inc, ticker_accion),
            use_container_width=True, theme="streamlit",
        )
        st.caption(
            "Flow shows how Revenue is distributed across Cost of Revenue, "
            "Operating Expenses, taxes & interest, and ultimately Net Income."
        )
    else:
        st.info("Insufficient income statement data for Sankey diagram.")

    # ── Cash Flow ────────────────────────────────────────────────────────────
    section_divider("Cash Flow Analysis")
    cf_col, struct_col = st.columns(2, gap="large")
    with cf_col:
        if cf.get("operating"):
            st.plotly_chart(
                plot_cashflow_bars(cf),
                use_container_width=True, theme="streamlit",
            )
        else:
            st.info("Cash flow data not available.")

    # ── Capital structure ────────────────────────────────────────────────────
    with struct_col:
        if bal.get("total_debt") and bal.get("equity"):
            st.plotly_chart(
                plot_capital_structure(bal),
                use_container_width=True, theme="streamlit",
            )
        else:
            st.info("Balance sheet data not available.")

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

    pe_fair_price = calculate_pe_fair_value(eps, sector_pe)
    ddm_price     = calculate_ddm(dividend, required_return, current_price)

    fmp_df = get_fmp_income_statement(ticker_accion, api_key)

    # Historical CAGR: drives both DCF short-term growth and Peter Lynch fair value
    if fmp_df is not None and "ebitda" in fmp_df.columns and eps:
        gr = calculate_5yr_cagr_from_fmp(fmp_df, "ebitda")
    else:
        _stmts_v = get_financial_statements(ticker_accion)
        _inc_v   = extract_income_data(_stmts_v.get("income"))
        _ni      = _inc_v.get("net_income") or _inc_v.get("operating_income")
        gr       = calc_cagr(_ni) if _ni else None

    # Revenue CAGR from yfinance income statement (DCF short-term growth driver)
    _stmts_dcf = get_financial_statements(ticker_accion)
    _inc_dcf   = extract_income_data(_stmts_dcf.get("income"))
    rev_cagr_v = calc_cagr(_inc_dcf.get("revenue"))

    dcf_price, annual_fcf, total_val, dcf_warn = calculate_dcf(
        fcf, shares_out, required_return, rev_cagr_v
    )

    if gr is not None and eps:
        lynch_price, lynch_caption, lynch_warn = calculate_lynch_fair_value(eps, gr)
    elif not eps:
        lynch_price, lynch_caption, lynch_warn = None, None, None
    else:
        lynch_price, lynch_caption, lynch_warn = None, None, None

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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top: 48px;
    padding: 16px 0 8px 0;
    border-top: 1px solid rgba(148,163,184,0.18);
    text-align: center;
    font-size: 0.75rem;
    opacity: 0.45;
    line-height: 1.8;
">
    This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
</div>
""", unsafe_allow_html=True)

