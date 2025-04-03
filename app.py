import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime
import requests

if __name__ == "__main__": 

    @st.cache_data(show_spinner=False)
    def get_industry_pe(ticker, api_key):
        try:
            # Step 1: Get industry info
            profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
            profile_resp = requests.get(profile_url).json()
            if not profile_resp:
                return None

            industry = profile_resp[0].get("industry")

            # Step 2: Get P/E by industry
            ratios_url = f"https://financialmodelingprep.com/api/v4/ratios-ttm-industry?apikey={api_key}"
            ratios_resp = requests.get(ratios_url).json()

            for entry in ratios_resp:
                if entry["industry"].lower() == industry.lower():
                    return entry.get("peRatioTTM", None)
        except Exception as e:
            st.warning(f"Error fetching industry P/E: {e}")
        return None

# Function to clean column names
    def clean_column_names(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.columns = [col.replace("(", "").replace(")", "").replace(",", "").strip() for col in df.columns]
        return df

# Function to download and clean data
    def download_ticker_data(ticker, start_date, end_date, interval):
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        data = clean_column_names(data)
        data.index = pd.to_datetime(data.index)

        if data.empty:
            st.error(f"No data found for ticker: {ticker}")
        return data


# Streamlit App Title
    st.title("CAPM Model Calculator")

# User inputs for stock selection
    st.sidebar.header("Select Stock & Market Index")
    ticker_accion = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
    ticker_indice = st.sidebar.selectbox("Select Market Index", ["^GSPC", "^IXIC", "^DJI"], index=0)
    interval = "1mo"

# Date selection
    start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
    today_date = datetime.date.today()

# Bond selection
    bond_ticker = st.sidebar.text_input("Enter Bond Ticker (e.g., ^TNX)", "^TNX")

# Download stock and index data
    data_accion = download_ticker_data(ticker_accion, start_date, today_date, interval)
    data_indice = download_ticker_data(ticker_indice, start_date, today_date, interval)

# Calculate Monthly Returns
    data_accion['Monthly_Return_Stock'] = data_accion["Close"].pct_change()
    data_indice['Monthly_Return_Index'] = data_indice["Close"].pct_change()

# Merge datasets
    data = data_accion.join(data_indice['Monthly_Return_Index']).dropna()

# Display Data
    tabs = st.tabs(["CAPM Model","Monthly Returns","Regression Analysis","Stock Data", "Market Data"])

# Calculate correlation and regression
    correlation = data['Monthly_Return_Stock'].corr(data['Monthly_Return_Index'])
    X = sm.add_constant(data['Monthly_Return_Index'])
    Y = data['Monthly_Return_Stock']
    lm = sm.OLS(Y, X).fit()
    intercept, beta_daily_return_indice = lm.params

    y_pred = beta_daily_return_indice * X['Monthly_Return_Index'] + intercept

# CAPM Calculation
    # Rf_data = download_ticker_data(bond_ticker, start_date, today_date, interval)
    # Rf = Rf_data["Close"].iloc[-1] / 100  # Convert percentage to decimal
    # Rm = ((1 + data['Monthly_Return_Index'].mean())**12) - 1  # Annualized Market Return
    # CAPM = (Rf * 100) + (beta_daily_return_indice * ((Rm - Rf) * 100))

    try:
        Rf_data = download_ticker_data(bond_ticker, start_date, today_date, interval)
        Rf = Rf_data["Close"].iloc[-1] / 100  # Convert to decimal
    except Exception as e:
        st.error(f"Could not fetch bond data for {bond_ticker}: {e}")
        Rf = 0.04 
    Rm = ((1 + data['Monthly_Return_Index'].mean())**12) - 1  # Annualized Market Return
    CAPM = (Rf * 100) + (beta_daily_return_indice * ((Rm - Rf) * 100))

    with tabs[0]:
        st.write("### CAPM Model Calculation")
        st.write(f"Risk-Free Rate ({bond_ticker}): {Rf*100:.2f}%")
        st.write(f"Expected Market Return: {Rm*100:.2f}%")
        st.write(f"Calculated CAPM Expected Return for {ticker_accion}: **{CAPM:.2f}%**")
    
        # Plot Security Market Line (SML)
        betas = np.linspace(0, 4, 20)
        expected_returns = Rf * 100 + betas * (Rm * 100 - Rf * 100)
    
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(betas, expected_returns, label="Security Market Line (SML)", color="blue", linewidth=2)
        ax.scatter(0, Rf * 100, color="red", marker="o", label="Risk-Free Rate")
        ax.scatter(1, Rm * 100, color="brown", marker="o", label="Market Return")
        ax.scatter(beta_daily_return_indice, CAPM, color="green", marker="o", s=100, label=f"{ticker_accion} (β={beta_daily_return_indice:.2f})")
    
        plt.xlabel("Beta (Systematic Risk)")
        plt.ylabel("Expected Return")
        plt.title("CAPM - Security Market Line")
        plt.axhline(y=Rf*100, color="gray", linestyle="--", linewidth=1)
        plt.axvline(x=1, color="gray", linestyle="--", linewidth=1, label="Market Beta = 1")
        plt.legend()
        plt.grid(False)
        st.pyplot(fig)

    with tabs[1]:
        st.write("### Monthly Returns Stock vs Index")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.lineplot(data=data, x=data.index, y='Monthly_Return_Stock', label='Monthly Return Stock')
        sns.lineplot(data=data, x=data.index, y='Monthly_Return_Index', label='Monthly Return Index')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.xlabel("Fecha")
        plt.ylabel("Retorno Mensual")
        plt.title("Monthly Returns Comparison")
        plt.legend()
        st.pyplot(fig)

    with tabs[2]:
        st.write("### Regression Analysis")
        st.write(f"Beta (β) of {ticker_accion}: {beta_daily_return_indice:.2f}")
        st.write(f"Correlation: {correlation:.2f}")
    
        # Scatterplot
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=X['Monthly_Return_Index'], y=Y, ax=ax)
        sns.lineplot(x=X['Monthly_Return_Index'], y=y_pred, color='red', ax=ax)
        plt.xlabel("Market Monthly Return")
        plt.ylabel("Stock Monthly Return")
        plt.title("Regression Analysis: Beta Calculation")
        st.pyplot(fig)

    with tabs[3]:
        st.write("### Stock Data")
        st.dataframe(data_accion[['Close', 'Monthly_Return_Stock']].dropna())

    with tabs[4]:
        st.write("### Market Index Data")
        st.dataframe(data_indice[['Close', 'Monthly_Return_Index']].dropna())


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

        # Get industry P/E from FMP
        api_key = st.secrets["fmp"]["api_key"]  
        industry_pe = get_industry_pe(ticker_accion, api_key)

        if not industry_pe or industry_pe <= 0:
            industry_pe = 25  # fallback value

        # Fair Price Calculations
        pe_fair_price = eps * industry_pe if eps else None

        ddm_price = None
        if dividend > 1 and required_return > dividend_growth:
            try:
                ddm_price = dividend * (1 + dividend_growth) / (required_return - dividend_growth)
            except ZeroDivisionError:
             pass

        dcf_price = None
        if fcf and fcf > 0 and shares_outstanding > 0:
            try:
                annual_fcf = fcf * 4

                # Adaptive logic
                if annual_fcf > 20_000_000_000:  # Mega cap
                    short_term_growth = 0.05
                    terminal_growth = 0.02
                elif annual_fcf > 5_000_000_000:  # Mature large-cap
                    short_term_growth = 0.06
                    terminal_growth = 0.025
                else:  # Smaller or growing company
                    short_term_growth = 0.10
                    terminal_growth = 0.03

                forecast_years = 5
                discount_rate = required_return

                # Project and discount FCF for forecast period
                fcf_list = []
                for year in range(1, forecast_years + 1):
                    projected_fcf = annual_fcf * (1 + short_term_growth) ** year
                    discounted_fcf = projected_fcf / (1 + discount_rate) ** year
                    fcf_list.append(discounted_fcf)

                # Terminal value
                final_fcf = annual_fcf * (1 + short_term_growth) ** forecast_years
                terminal_value = final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
                discounted_terminal_value = terminal_value / (1 + discount_rate) ** forecast_years

                total_value = sum(fcf_list) + discounted_terminal_value
                dcf_price = total_value / shares_outstanding
            except ZeroDivisionError:
             pass


        st.write(f"**Current Price:** ${current_price:.2f}" if current_price else "No price available.")

        st.subheader("📊 Estimated Fair Values:")
        if pe_fair_price:
            st.write(f"**P/E Method:** ${pe_fair_price:.2f}")
        else:
            st.write("P/E Method: Not enough data.")

        if ddm_price:
            st.write(f"**DDM Method:** ${ddm_price:.2f}")
        else:
            st.write("DDM Method: Not applicable (low or no dividend).")

        if dcf_price:
            st.write(f"**DCF Method:** ${dcf_price:.2f}")
        else:
         st.write("DCF Method: Not enough data.")

    # Optional debug for transparency
    with st.expander("🔍 Show Raw Inputs"):
        st.write("EPS:", eps)
        st.write("Industry P/E (used):", industry_pe)
        st.write("Dividend Rate:", dividend)
        st.write("Required Return (CAPM):", required_return)
        st.write("Free Cash Flow (Annualized):", fcf * 4 if fcf else None)
        st.write("Shares Outstanding:", shares_outstanding)








