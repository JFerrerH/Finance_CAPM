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
        profile_resp = requests.get(profile_url).json()

        if not profile_resp or not isinstance(profile_resp, list):
            st.warning("⚠️ No profile data returned.")
            return None

        sector = profile_resp[0].get("sector")
        st.write("Sector:", sector)

        if not sector:
            return None

        sector_url = f"https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?apikey={api_key}"
        sector_resp = requests.get(sector_url).json()

        if isinstance(sector_resp, dict) and "Error Message" in sector_resp:
            return SECTOR_PE_FALLBACK.get(sector, 25)

        if not isinstance(sector_resp, list):
            st.warning("⚠️ Unexpected response structure for sector P/E.")
            return SECTOR_PE_FALLBACK.get(sector, 25)

        for entry in sector_resp:
            if entry.get("sector", "").lower() == sector.lower():
                return entry.get("peRatioTTM")

        st.warning("⚠️ Sector not found in P/E list. Using fallback.")
        return SECTOR_PE_FALLBACK.get(sector, 25)

    except Exception as e:
        st.warning(f"Error fetching sector P/E: {e}")
        return None


def get_fmp_income_statement(ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=5&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        return df[["date", "revenue", "ebitda"]].dropna()
    except Exception as e:
        st.warning(f"Error fetching FMP financials: {e}")
        return None
