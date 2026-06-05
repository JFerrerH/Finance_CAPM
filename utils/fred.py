"""
utils/fred.py
-------------
Fetches real economic data from the St. Louis Fed (FRED) public API.

Short-cycle series (used in geopolitical filter and economic fundamentals):
    CPIAUCSL  — CPI All Urban Consumers, All Items (headline inflation)
    CPILFESL  — CPI ex Food & Energy (core inflation)
    UNRATE    — Civilian Unemployment Rate
    T5YIE     — 5-Year Breakeven Inflation Rate
    T10YIE    — 10-Year Breakeven Inflation Rate
    INDPRO    — Industrial Production Index (manufacturing activity proxy)

Big Cycle series (Dalio long-term debt cycle positioning):
    GFDEGDQ188S — Federal Debt as % of GDP (quarterly)
    M2SL        — M2 Money Supply (monthly)
    FEDFUNDS    — Effective Federal Funds Rate (monthly)

Requires a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
Add to .streamlit/secrets.toml:
    [fred]
    api_key = "your_key_here"
"""

import requests
import pandas as pd
import streamlit as st


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# (series_id, frequency): 'm' = monthly, 'd' = daily, 'q' = quarterly
# Daily and quarterly series are resampled to month-end for consistency.
FRED_SERIES: dict[str, tuple[str, str]] = {
    # ── Short-cycle: the two canonical Dalio axes ─────────────────────────────
    "Real GDP":        ("GDPC1",         "q"),   # Real GDP — the actual growth measure
    "CPI Headline":    ("CPIAUCSL",      "m"),   # Headline CPI — the actual inflation measure
    "CPI Core":        ("CPILFESL",      "m"),   # Core CPI (supplementary, geo filter)
    "Unemployment":    ("UNRATE",        "m"),   # Labour market health (supplementary)
    "TIPS 5Y":         ("T5YIE",         "d"),   # Inflation expectations (geo filter)
    "TIPS 10Y":        ("T10YIE",        "d"),
    "Industrial Prod": ("INDPRO",        "m"),   # Manufacturing activity (supplementary)
    # ── Big Cycle (Dalio long-term debt cycle) ────────────────────────────────
    "Debt/GDP":        ("GFDEGDQ188S",   "q"),   # Federal Debt as % of GDP
    "M2":              ("M2SL",          "m"),   # M2 Money Supply
    "Fed Funds":       ("FEDFUNDS",      "m"),   # Effective Federal Funds Rate
}


@st.cache_data(ttl=86_400 * 7, show_spinner=False)
def get_big_cycle_history(api_key: str) -> dict:
    """
    Fetches long-term Debt/GDP and Fed Funds from 1966 onward for the Big Cycle
    history chart. Cached weekly — these series change slowly.
    Returns {name: pd.Series} or empty dict if unavailable.
    """
    if not api_key:
        return {}
    today = str(pd.Timestamp.today().date())
    series = {
        "Debt/GDP":  ("GFDEGDQ188S", "q", "1966-01-01"),
        "Fed Funds": ("FEDFUNDS",    "m", "1966-01-01"),
    }
    result: dict = {}
    for name, (sid, freq, start) in series.items():
        try:
            resp = requests.get(
                FRED_BASE,
                params={"series_id": sid, "observation_start": start,
                        "observation_end": today, "frequency": freq,
                        "api_key": api_key, "file_type": "json"},
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
            s = pd.DataFrame(records).set_index("date")["value"]
            if freq in ("d", "q"):
                s = s.resample("ME").last()
            result[name] = s.dropna()
        except Exception:
            pass
    return result


@st.cache_data(ttl=86_400, show_spinner=False)
def get_economic_data(api_key: str, start: str, end: str,
                      series_version: str = "") -> dict:
    """
    Fetches each FRED series and returns {name: pd.Series} indexed by month-end dates.
    Returns an empty dict when the api_key is blank or every fetch fails —
    the rest of the app degrades gracefully to market-price-only signals.

    series_version is a hash of FRED_SERIES keys — changing it busts the cache
    so new series (like GDPC1) are fetched immediately without manual cache clear.
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
            if freq in ("d", "q"):   # daily and quarterly both resample to month-end
                series = series.resample("ME").last()
            result[name] = series.dropna()

        except Exception:
            pass

    return result
