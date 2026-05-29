import requests
import pandas as pd
import streamlit as st


SECTOR_PE_FALLBACK = {
    "Technology": 25,
    "Consumer Defensive": 22,
    "Healthcare": 24,
    "Financial Services": 14,
    "Energy": 12,
    "Industrials": 17,
    "Consumer Cyclical": 20,
    "Communication Services": 18,
    "Utilities": 16,
    "Real Estate": 19,
    "Materials": 15,
}


@st.cache_data(show_spinner=False)
def get_sector_pe(ticker, api_key):
    try:
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
        profile_resp = requests.get(profile_url, timeout=10).json()

        if not profile_resp or not isinstance(profile_resp, list):
            return None

        sector = profile_resp[0].get("sector")
        if not sector:
            return None

        sector_url = f"https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?apikey={api_key}"
        sector_resp = requests.get(sector_url, timeout=10).json()

        if isinstance(sector_resp, dict) and "Error Message" in sector_resp:
            return SECTOR_PE_FALLBACK.get(sector, 25)

        if not isinstance(sector_resp, list):
            return SECTOR_PE_FALLBACK.get(sector, 25)

        for entry in sector_resp:
            if entry.get("sector", "").lower() == sector.lower():
                return entry.get("peRatioTTM")

        return SECTOR_PE_FALLBACK.get(sector, 25)

    except Exception:
        return None


def get_fmp_income_statement(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=5&apikey={api_key}"
    try:
        data = requests.get(url, timeout=10).json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        df = pd.DataFrame(data)
        cols = [c for c in ["date", "revenue", "ebitda"] if c in df.columns]
        if len(cols) < 2:
            return None
        return df[cols].dropna()
    except Exception:
        return None
