import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime
import requests

# === Cached function to fetch Sector P/E from FMP ===

@st.cache_data(show_spinner=False)
def get_sector_pe(ticker, api_key):
    try:
        # Get sector from company profile
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
        profile_resp = requests.get(profile_url).json()
        if not profile_resp or not isinstance(profile_resp, list):
            return None

        sector = profile_resp[0].get("sector")
        if not sector:
            return None

        # Get sector-level P/E ratios
        sector_url = f"https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?apikey={api_key}"
        sector_resp = requests.get(sector_url).json()
        if not isinstance(sector_resp, list):
            return None

        for entry in sector_resp:
            if entry.get("sector") == sector:
                return entry.get("peRatioTTM", None)
    
    except Exception as e:
        st.warning(f"Error fetching sector P/E: {e}")
    return None

# === CAGR 5 yr ===
def calculate_5yr_ebitda_cagr(ticker):
    stock = yf.Ticker(ticker)
    financials = stock.financials
    ebitda = financials.loc['EBITDA']
    
    if len(ebitda) >= 5:
        ebitda_5yrs_ago = ebitda[-1]
        ebitda_current = ebitda[0]
        cagr = (ebitda_current / ebitda_5yrs_ago) ** (1/5) - 1
        return cagr
    else:
        return None
# === Clean and download functions ===
def clean_column_names(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [col.replace("(", "").replace(")", "").replace(",", "").strip() for col in df.columns]
    return df

def download_ticker_data(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data = clean_column_names(data)
    data.index = pd.to_datetime(data.index)
    if data.empty:
        st.error(f"No data found for ticker: {ticker}")
    return data

# === UI + Data Loading ===
st.title("CAPM Model Calculator")

st.sidebar.header("Select Stock & Market Index")
ticker_accion = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
ticker_indice = st.sidebar.selectbox("Select Market Index", ["^GSPC", "^IXIC", "^DJI"], index=0)
interval = "1mo"
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
today_date = datetime.date.today()
bond_ticker = st.sidebar.text_input("Enter Bond Ticker (e.g., ^TNX)", "^TNX")

data_accion = download_ticker_data(ticker_accion, start_date, today_date, interval)
data_indice = download_ticker_data(ticker_indice, start_date, today_date, interval)

data_accion['Monthly_Return_Stock'] = data_accion["Close"].pct_change()
data_indice['Monthly_Return_Index'] = data_indice["Close"].pct_change()

data = data_accion.join(data_indice['Monthly_Return_Index']).dropna()

tabs = st.tabs(["CAPM Model", "Monthly Returns", "Regression Analysis", "Stock Data", "Market Data"])

correlation = data['Monthly_Return_Stock'].corr(data['Monthly_Return_Index'])
X = sm.add_constant(data['Monthly_Return_Index'])
Y = data['Monthly_Return_Stock']
lm = sm.OLS(Y, X).fit()
intercept, beta_daily_return_indice = lm.params
y_pred = beta_daily_return_indice * X['Monthly_Return_Index'] + intercept

try:
    Rf_data = download_ticker_data(bond_ticker, start_date, today_date, interval)
    Rf = Rf_data["Close"].iloc[-1] / 100
except Exception as e:
    st.error(f"Could not fetch bond data for {bond_ticker}: {e}")
    Rf = 0.04
Rm = ((1 + data['Monthly_Return_Index'].mean()) ** 12) - 1
CAPM = (Rf * 100) + (beta_daily_return_indice * ((Rm - Rf) * 100))

# === CAPM Tab ===
with tabs[0]:
    st.write("### CAPM Model Calculation")
    st.write(f"Risk-Free Rate ({bond_ticker}): {Rf*100:.2f}%")
    st.write(f"Expected Market Return: {Rm*100:.2f}%")
    st.write(f"Calculated CAPM Expected Return for {ticker_accion}: **{CAPM:.2f}%**")

    betas = np.linspace(0, 4, 20)
    expected_returns = Rf * 100 + betas * (Rm * 100 - Rf * 100)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(betas, expected_returns, label="Security Market Line (SML)", color="blue", linewidth=2)
    ax.scatter(0, Rf * 100, color="red", marker="o", label="Risk-Free Rate")
    ax.scatter(1, Rm * 100, color="brown", marker="o", label="Market Return")
    ax.scatter(beta_daily_return_indice, CAPM, color="green", marker="o", s=100, label=f"{ticker_accion} (β={beta_daily_return_indice:.2f})")

    plt.xlabel("Beta (Systematic Risk)")
    plt.ylabel("Expected Return")
    plt.title("CAPM - Security Market Line")
    plt.axhline(y=Rf * 100, color="gray", linestyle="--", linewidth=1)
    plt.axvline(x=1, color="gray", linestyle="--", linewidth=1, label="Market Beta = 1")
    plt.legend()
    plt.grid(False)
    st.pyplot(fig)

# === Monthly Returns ===
with tabs[1]:
    st.write("### Monthly Returns Stock vs Index")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=data, x=data.index, y='Monthly_Return_Stock', label='Monthly Return Stock')
    sns.lineplot(data=data, x=data.index, y='Monthly_Return_Index', label='Monthly Return Index')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xlabel("Date")
    plt.ylabel("Monthly Return")
    plt.title("Monthly Returns Comparison")
    plt.legend()
    st.pyplot(fig)

# === Regression ===
with tabs[2]:
    st.write("### Regression Analysis")
    st.write(f"Beta (β) of {ticker_accion}: {beta_daily_return_indice:.2f}")
    st.write(f"Correlation: {correlation:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=X['Monthly_Return_Index'], y=Y, ax=ax)
    sns.lineplot(x=X['Monthly_Return_Index'], y=y_pred, color='red', ax=ax)
    plt.xlabel("Market Monthly Return")
    plt.ylabel("Stock Monthly Return")
    plt.title("Regression Analysis: Beta Calculation")
    st.pyplot(fig)

# === Stock Data ===
with tabs[3]:
    st.write("### Stock Data")
    st.dataframe(data_accion[['Close', 'Monthly_Return_Stock']].dropna())

# === Market Data ===
with tabs[4]:
    st.write("### Market Index Data")
    st.dataframe(data_indice[['Close', 'Monthly_Return_Index']].dropna())

# === Fair Value Estimations ===
with st.tabs(["Fair Value Estimations"])[0]:
    st.write("### Fair Price Estimations for", ticker_accion.upper())

    stock = yf.Ticker(ticker_accion)
    info = stock.get_info()

    current_price = info.get("currentPrice")
    eps = info.get("trailingEps")
    pe_ratio = info.get("trailingPE")
    dividend = info.get("dividendRate", 0) or 0
    dividend_growth = 0.05
    required_return = CAPM / 100 if CAPM else 0.10
    fcf = info.get("freeCashflow", 0)
    shares_outstanding = info.get("sharesOutstanding", 1)

    api_key = st.secrets["fmp"]["api_key"]
    sector_pe = get_sector_pe(ticker_accion, api_key)
    if not sector_pe or sector_pe <= 0:
        sector_pe = 25

    pe_fair_price = eps * sector_pe if eps else None

    ddm_price = None
    if dividend > 1 and required_return > dividend_growth:
        try:
            ddm_price = dividend * (1 + dividend_growth) / (required_return - dividend_growth)
        except ZeroDivisionError:
            pass

    dcf_price = None
    if fcf and fcf > 0 and shares_outstanding > 0:
        try:
            annual_fcf = fcf

            # ⚠️ Sanity check for abnormally high FCF
            if annual_fcf > 80_000_000_000:
                st.warning("⚠️ FCF seems abnormally high — this may affect DCF accuracy.")

            if annual_fcf > 20_000_000_000:
                short_term_growth = 0.05
                terminal_growth = 0.02
            elif annual_fcf > 5_000_000_000:
                short_term_growth = 0.06
                terminal_growth = 0.025
            else:
                short_term_growth = 0.10
                terminal_growth = 0.03

            forecast_years = 5
            discount_rate = required_return

            fcf_list = []
            for year in range(1, forecast_years + 1):
                projected_fcf = annual_fcf * (1 + short_term_growth) ** year
                discounted_fcf = projected_fcf / (1 + discount_rate) ** year
                fcf_list.append(discounted_fcf)

            final_fcf = annual_fcf * (1 + short_term_growth) ** forecast_years
            terminal_value = final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
            discounted_terminal_value = terminal_value / (1 + discount_rate) ** forecast_years

            total_value = sum(fcf_list) + discounted_terminal_value
            dcf_price = total_value / shares_outstanding

            st.write("Annual FCF:", annual_fcf)
            st.write("Total Firm Value:", total_value)

        except Exception as e:
            st.warning(f"⚠️ DCF calculation failed: {e}")
        
    # === Peter Lynch Fair Value Calculation ===
    growth_rate = calculate_5yr_ebitda_cagr(ticker_accion)
    lynch_fair_value = None

    st.write("🔍 EPS:", eps)
    st.write("🔍 Growth rate:", growth_rate)

    if eps and growth_rate is not None:
        if growth_rate > 0.25:
            growth_rate = 0.25
        elif growth_rate < 0.05:
            growth_rate = None  # Too low, method not applicable

        if growth_rate is not None:
            lynch_fair_value = eps * growth_rate * 100  # PEG = 1 → P/E = Growth%
            st.write(f"**Peter Lynch Fair Value:** ${lynch_fair_value:.2f}")
        else:
            st.write("Peter Lynch Method: Not applicable (growth rate out of range).")
    else:
        st.write("Insufficient data to calculate Peter Lynch Fair Value.")

       
    st.write(f"**Current Price:** ${current_price:.2f}" if current_price else "No price available.")
    st.subheader("📊 Estimated Fair Values:")
    st.write(f"**P/E Method:** ${pe_fair_price:.2f}" if pe_fair_price else "P/E Method: Not enough data.")
    st.write(f"**DDM Method:** ${ddm_price:.2f}" if ddm_price else "DDM Method: Not applicable.")
    st.write(f"**DCF Method:** ${dcf_price:.2f}" if dcf_price else "DCF Method: Not enough data.")

    with st.expander("🔍 Show Raw Inputs"):
        st.write("EPS:", eps)
        st.write("Sector P/E (used):", sector_pe)
        st.write("Dividend Rate:", dividend)
        st.write("Required Return (CAPM):", required_return)
        st.write("Free Cash Flow (Annualized):", fcf if fcf else None)
        st.write("Shares Outstanding:", shares_outstanding)
    