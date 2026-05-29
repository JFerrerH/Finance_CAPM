"""
utils/financials.py
-------------------
Fetches and normalises annual financial statements from yfinance.
Provides three extractors (income, cash flow, balance sheet) that
return plain dicts of {label: pd.Series(values, index=years)}.
All monetary values are returned in the original units (USD).
"""

import pandas as pd
import streamlit as st
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_statements(ticker: str) -> dict:
    """
    Returns {"income": df, "balance": df, "cashflow": df}.
    DataFrames have line-item rows and date columns (most recent first).
    Returns empty dicts on failure — callers must guard with .get().
    """
    try:
        t = yf.Ticker(ticker)
        return {
            "income":   t.financials,
            "balance":  t.balance_sheet,
            "cashflow": t.cashflow,
        }
    except Exception:
        return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_row(df: pd.DataFrame, *keys) -> pd.Series | None:
    """
    Case-insensitive fuzzy row lookup. Tries each key in order;
    returns the first match as a Series, or None.
    """
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for key in keys:
        kl = key.lower()
        # exact match first
        if kl in idx_lower:
            return df.loc[idx_lower[kl]]
        # substring match
        for name_lower, name_real in idx_lower.items():
            if kl in name_lower:
                return df.loc[name_real]
    return None


def _trim(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Keep only the n most recent columns (most-recent-first ordering)."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.iloc[:, : min(n, df.shape[1])]


def _year_labels(series: pd.Series) -> list[str]:
    """Convert DatetimeIndex to year strings."""
    try:
        return [str(c.year) for c in series.index]
    except Exception:
        return [str(c)[:4] for c in series.index]


# ─────────────────────────────────────────────────────────────────────────────
# Extractors
# ─────────────────────────────────────────────────────────────────────────────

def extract_income_data(income_df: pd.DataFrame) -> dict:
    """
    Returns dict with keys:
        revenue, cost_of_revenue, gross_profit, rd, sga,
        operating_income, pretax_income, tax, net_income, years
    All values are lists (oldest → newest) or None.
    """
    df = _trim(income_df)
    if df.empty:
        return {}

    def _vals(series):
        if series is None:
            return None
        # reverse so oldest year is index 0
        return list(reversed([float(v) if pd.notna(v) else 0.0 for v in series.values]))

    def _years(series):
        if series is None:
            return []
        return list(reversed(_year_labels(series)))

    revenue          = _find_row(df, "Total Revenue", "Revenue")
    cost_of_revenue  = _find_row(df, "Cost Of Revenue", "Reconciled Cost Of Revenue")
    gross_profit     = _find_row(df, "Gross Profit")
    rd               = _find_row(df, "Research And Development", "R&D Expenses")
    sga              = _find_row(df, "Selling General And Administrative", "SG&A")
    operating_income = _find_row(df, "Operating Income", "EBIT", "Total Operating Income")
    pretax_income    = _find_row(df, "Pretax Income", "Income Before Tax")
    tax              = _find_row(df, "Tax Provision", "Income Tax Expense")
    net_income       = _find_row(df, "Net Income", "Net Income Common Stockholders")

    ref = revenue if revenue is not None else net_income
    return {
        "revenue":          _vals(revenue),
        "cost_of_revenue":  _vals(cost_of_revenue),
        "gross_profit":     _vals(gross_profit),
        "rd":               _vals(rd),
        "sga":              _vals(sga),
        "operating_income": _vals(operating_income),
        "pretax_income":    _vals(pretax_income),
        "tax":              _vals(tax),
        "net_income":       _vals(net_income),
        "years":            _years(ref),
    }


def extract_cashflow_data(cf_df: pd.DataFrame) -> dict:
    df = _trim(cf_df)
    if df.empty:
        return {}

    def _vals(series):
        if series is None:
            return None
        return list(reversed([float(v) if pd.notna(v) else 0.0 for v in series.values]))

    operating = _find_row(df, "Operating Cash Flow", "Cash From Operations",
                          "Total Cash From Operating Activities",
                          "Cash Flow From Continuing Operating Activities")
    investing = _find_row(df, "Investing Cash Flow", "Cash From Investing",
                          "Total Cash From Investing Activities",
                          "Cash Flow From Continuing Investing Activities")
    financing = _find_row(df, "Financing Cash Flow", "Cash From Financing",
                          "Total Cash From Financing Activities",
                          "Cash Flow From Continuing Financing Activities")
    fcf       = _find_row(df, "Free Cash Flow")
    capex     = _find_row(df, "Capital Expenditure", "Capital Expenditures",
                          "Purchases Of Property Plant And Equipment")

    ref = operating if operating is not None else investing
    years = list(reversed(_year_labels(ref))) if ref is not None else []

    # Derive FCF from OCF - |Capex| if not directly available
    if fcf is None and operating is not None and capex is not None:
        op_v  = [float(v) if pd.notna(v) else 0.0 for v in operating.values]
        cx_v  = [float(v) if pd.notna(v) else 0.0 for v in capex.values]
        fcf_v = [o + c for o, c in zip(op_v, cx_v)]  # capex is usually negative
        fcf   = pd.Series(fcf_v, index=operating.index)

    return {
        "operating": _vals(operating),
        "investing": _vals(investing),
        "financing": _vals(financing),
        "fcf":       _vals(fcf),
        "capex":     _vals(capex),
        "years":     years,
    }


def extract_balance_data(bal_df: pd.DataFrame) -> dict:
    df = _trim(bal_df)
    if df.empty:
        return {}

    def _vals(series):
        if series is None:
            return None
        return list(reversed([float(v) if pd.notna(v) else 0.0 for v in series.values]))

    total_assets  = _find_row(df, "Total Assets")
    total_liab    = _find_row(df, "Total Liabilities Net Minority Interest",
                               "Total Liabilities", "Total Liab")
    equity        = _find_row(df, "Stockholders Equity", "Total Equity",
                               "Common Stock Equity", "Stockholder Equity")
    total_debt    = _find_row(df, "Total Debt")
    cash          = _find_row(df, "Cash And Cash Equivalents",
                               "Cash Cash Equivalents And Short Term Investments", "Cash")
    curr_assets   = _find_row(df, "Current Assets", "Total Current Assets")
    curr_liab     = _find_row(df, "Current Liabilities", "Total Current Liabilities")

    ref = total_assets if total_assets is not None else equity
    years = list(reversed(_year_labels(ref))) if ref is not None else []

    return {
        "total_assets":  _vals(total_assets),
        "total_liab":    _vals(total_liab),
        "equity":        _vals(equity),
        "total_debt":    _vals(total_debt),
        "cash":          _vals(cash),
        "curr_assets":   _vals(curr_assets),
        "curr_liab":     _vals(curr_liab),
        "years":         years,
    }


# ─────────────────────────────────────────────────────────────────────────────
# KPI helpers
# ─────────────────────────────────────────────────────────────────────────────

def yoy_pct(values: list | None) -> float | None:
    """Return YoY % change between the last two values in the list."""
    if not values or len(values) < 2:
        return None
    prev, curr = values[-2], values[-1]
    if prev == 0:
        return None
    return (curr - prev) / abs(prev)


def fmt_large(val: float | None) -> str:
    """Format a large USD number as $XB or $XM."""
    if val is None:
        return "N/A"
    abs_v = abs(val)
    sign  = "-" if val < 0 else ""
    if abs_v >= 1e12:
        return f"{sign}${abs_v/1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{sign}${abs_v/1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{sign}${abs_v/1e6:.1f}M"
    return f"{sign}${abs_v:,.0f}"


def calc_cagr(values: list) -> float | None:
    """
    Compound Annual Growth Rate from a list of annual values (oldest → newest).
    Returns None if fewer than 2 valid positive values are present.
    """
    valid = [float(v) for v in (values or []) if v and float(v) > 0]
    if len(valid) < 2:
        return None
    n = len(valid) - 1
    return (valid[-1] / valid[0]) ** (1 / n) - 1
