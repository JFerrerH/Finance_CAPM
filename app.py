import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime

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
    return data

# Streamlit App Title
st.title("CAPM Model Calculator")

# User inputs for stock selection
st.sidebar.header("Select Stock & Market Index")
ticker_accion = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
ticker_indice = st.sidebar.selectbox("Select Market Index", ["^GSPC", "^IXIC", "^DJI"], index=0)
interval = '1mo'

# Date selection
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
today_date = datetime.date.today()

# Download stock and index data
data_accion = download_ticker_data(ticker_accion, start_date, today_date, interval)
data_indice = download_ticker_data(ticker_indice, start_date, today_date, interval)

# Calculate Monthly Returns
data_accion['Monthly_Return_Stock'] = data_accion["Close"].pct_change()
data_indice['Monthly_Return_Index'] = data_indice["Close"].pct_change()

# Merge datasets
data = data_accion.join(data_indice['Monthly_Return_Index']).dropna()

# Display Data
tabs = st.tabs(["CAPM Model","Regression Analysis","Stock Data", "Market Data"])

with tabs[2]:
    st.write("### Stock Data")
    st.dataframe(data_accion[['Close', 'Monthly_Return_Stock']].dropna())

with tabs[3]:
    st.write("### Market Index Data")
    st.dataframe(data_indice[['Close', 'Monthly_Return_Index']].dropna())

# Calculate correlation and regression
correlation = data['Monthly_Return_Stock'].corr(data['Monthly_Return_Index'])
X = sm.add_constant(data['Monthly_Return_Index'])
Y = data['Monthly_Return_Stock']
lm = sm.OLS(Y, X).fit()
intercept, beta_daily_return_indice = lm.params

y_pred = beta_daily_return_indice * X['Monthly_Return_Index'] + intercept

with tabs[1]:
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

# CAPM Calculation
bond_ticker = st.sidebar.text_input("Enter Bond Ticker (e.g., ^TNX)", "^TNX")
Rf_data = download_ticker_data(bond_ticker, start_date, today_date, interval)
Rf = Rf_data["Close"].iloc[-1] / 100  # Convert percentage to decimal
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


