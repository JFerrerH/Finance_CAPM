"""
utils/fred.py
-------------
Fetches real economic data from the St. Louis Fed (FRED) public API.

Series fetched:
    CPIAUCSL  — CPI All Urban Consumers, All Items (headline inflation)
    CPILFESL  — CPI ex Food & Energy (core inflation — structural vs supply-shock discriminator)
    UNRATE    — Civilian Unemployment Rate
    T5YIE     — 5-Year Breakeven Inflation Rate (market-implied inflation expectations)
    T10YIE    — 10-Year Breakeven Inflation Rate
    INDPRO    — Industrial Production Index (manufacturing activity proxy)

Requires a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
Add to .streamlit/secrets.toml:
    [fred]
    api_key = "your_key_here"
"""

import requests
import pandas as pd
import streamlit as st


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# (series_id, frequency): 'm' = monthly, 'd' = daily (resampled to month-end)
FRED_SERIES: dict[str, tuple[str, str]] = {
    "CPI Headline":    ("CPIAUCSL", "m"),
    "CPI Core":        ("CPILFESL", "m"),
    "Unemployment":    ("UNRATE",   "m"),
    "TIPS 5Y":         ("T5YIE",    "d"),
    "TIPS 10Y":        ("T10YIE",   "d"),
    "Industrial Prod": ("INDPRO",   "m"),
}


@st.cache_data(ttl=86_400, show_spinner=False)
def get_economic_data(api_key: str, start: str, end: str) -> dict:
    """
    Fetches each FRED series and returns {name: pd.Series} indexed by month-end dates.
    Returns an empty dict when the api_key is blank or every fetch fails —
    the rest of the app degrades gracefully to market-price-only signals.
    """
    if not api_key:
        return {}

    result: dict = {}
    for name, (series_id, freq) in FRED_SERIES.items():
        try:
            resp = requests.get(
                FRED_BASE,
                params={
                    "series_id":         series_id,
                    "observation_start": start,
                    "observation_end":   end,
                    "frequency":         freq,
                    "api_key":           api_key,
                    "file_type":         "json",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                continue

            records = [
                {"date": pd.to_datetime(obs["date"]), "value": float(obs["value"])}
                for obs in resp.json().get("observations", [])
                if obs.get("value") not in (None, ".")
            ]
            if not records:
                continue

            series = pd.DataFrame(records).set_index("date")["value"]
            if freq == "d":
                series = series.resample("ME").last()
            result[name] = series.dropna()

        except Exception:
            pass

    return result
