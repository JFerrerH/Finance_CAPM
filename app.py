import streamlit as st
import datetime

from utils.data import download_ticker_data, get_stock_info
from utils.fmp import get_sector_pe, get_fmp_income_statement
from utils.capm import calculate_capm
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
)

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
        ["📊  Dashboard", "📉  Analysis", "💰  Valuation", "🗃  Raw Data"],
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
        stat_card("Risk-Free Rate", f"{Rf * 100:.2f}%")
        stat_card("Mkt. Return (Rm)", f"{Rm * 100:.2f}%")

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

