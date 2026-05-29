import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_info(ticker):
    try:
        return yf.Ticker(ticker).get_info()
    except Exception:
        return {}


def clean_column_names(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [
        col.replace("(", "").replace(")", "").replace(",", "").strip()
        for col in df.columns
    ]
    return df


def download_ticker_data(ticker, start_date, end_date, interval, silent=False):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data = clean_column_names(data)
    data.index = pd.to_datetime(data.index)
    if data.empty and not silent:
        st.error(f"No data found for ticker: {ticker}")
    return data
