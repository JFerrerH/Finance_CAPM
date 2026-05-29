"""
utils/macro.py
--------------
Downloads and interprets macro-economic indicators for
Ray Dalio's cycle framework (Big / Short-term debt cycles).

Data sources: yfinance (free, no extra API key required).
Indicators used:
    ^TNX   — US 10-Year Treasury yield
    ^VIX   — CBOE Volatility Index (fear gauge)
    DX-Y.NYB — US Dollar Index
    GC=F   — Gold (inflation / safe-haven proxy)
    ^GSPC  — S&P 500 (growth proxy)
    CL=F   — Crude Oil (commodity / inflation proxy)
"""

import pandas as pd
import yfinance as yf
import streamlit as st
from utils.data import download_ticker_data


# ─────────────────────────────────────────────────────────────────────────────
# Ticker map
# ─────────────────────────────────────────────────────────────────────────────

MACRO_TICKERS = {
    "10Y Yield":    "^TNX",
    "VIX":          "^VIX",
    "Dollar":       "DX-Y.NYB",
    "Gold":         "GC=F",
    "S&P 500":      "^GSPC",
    "Oil":          "CL=F",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_data(start, end) -> dict:
    """
    Returns {name: pd.Series(Close, monthly)} for each macro indicator.
    Missing tickers are silently excluded.
    For index tickers (^TNX, ^VIX, etc.) monthly data can be sparse;
    falls back to daily data resampled to month-end.
    """
    result = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            # Try monthly first
            df = download_ticker_data(ticker, start, end, "1mo", silent=True)
            if df is None or df.empty or "Close" not in df.columns:
                # Fallback: daily → resample to month-end
                df = download_ticker_data(ticker, start, end, "1d", silent=True)
                if df is not None and not df.empty and "Close" in df.columns:
                    df = df[["Close"]].resample("ME").last()
            if df is not None and not df.empty and "Close" in df.columns:
                s = df["Close"].dropna()
                if not s.empty:
                    result[name] = s
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(series: pd.Series, n: int) -> float | None:
    """n-month % change at the tail of the series."""
    if series is None or len(series) < n + 1:
        return None
    return float(series.iloc[-1]) / float(series.iloc[-1 - n]) - 1


def _abs(series: pd.Series, n: int) -> float | None:
    """n-month absolute change at the tail of the series."""
    if series is None or len(series) < n + 1:
        return None
    return float(series.iloc[-1]) - float(series.iloc[-1 - n])


# ─────────────────────────────────────────────────────────────────────────────
# Signal computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cycle_signals(macro_data: dict) -> dict:
    """
    Computes Dalio-inspired cycle signals from the macro data dict.

    Returns a flat dict with:
        rate_current, rate_3m_bp, rate_direction
        vix_current, vix_label, vix_color
        sp_6m, growth_signal, growth_score   (-1 … +1 for quadrant)
        gold_6m, inflation_signal, inflation_score   (-1 … +1)
        dxy_3m, dxy_direction
        oil_3m
        cycle_phase, cycle_icon, cycle_color, cycle_desc
        cycle_stocks, cycle_bonds
    """
    s: dict = {}

    # ── Interest rates ────────────────────────────────────────────────────────
    tnx = macro_data.get("10Y Yield")
    if tnx is not None and len(tnx) >= 4:
        s["rate_current"]  = round(float(tnx.iloc[-1]), 2)
        chg3               = _abs(tnx, 3) or 0
        s["rate_3m_bp"]    = round(chg3 * 100)          # basis points
        s["rate_direction"] = (
            "Rising ↑"  if chg3 > 0.15 else
            "Falling ↓" if chg3 < -0.15 else
            "Stable →"
        )
    else:
        s.update(rate_current=None, rate_3m_bp=None, rate_direction="—")

    # ── Fear (VIX) ────────────────────────────────────────────────────────────
    vix = macro_data.get("VIX")
    if vix is not None and not vix.empty:
        vv = float(vix.iloc[-1])
        s["vix_current"] = round(vv, 1)
        if vv < 15:
            s["vix_label"], s["vix_color"] = "Calm (<15)",        "#10b981"
        elif vv < 25:
            s["vix_label"], s["vix_color"] = "Uncertain (15-25)", "#f59e0b"
        else:
            s["vix_label"], s["vix_color"] = "Fear (>25)",        "#ef4444"
    else:
        s.update(vix_current=None, vix_label="—", vix_color="#94a3b8")

    # ── Growth (S&P 500 proxy) ────────────────────────────────────────────────
    sp = macro_data.get("S&P 500")
    if sp is not None and len(sp) >= 7:
        sp_6m = _pct(sp, 6) or 0
        s["sp_6m"]          = round(sp_6m * 100, 1)
        s["sp_12m"]         = round((_pct(sp, min(12, len(sp) - 1)) or 0) * 100, 1)
        s["growth_signal"]  = (
            "Expanding"   if sp_6m >  0.03 else
            "Contracting" if sp_6m < -0.03 else
            "Flat"
        )
        s["growth_score"]   = max(-1.0, min(1.0, sp_6m * 5))   # normalised
    else:
        s.update(sp_6m=None, sp_12m=None, growth_signal="—", growth_score=0.0)

    # ── Inflation (Gold + rate level proxy) ──────────────────────────────────
    gold = macro_data.get("Gold")
    if gold is not None and len(gold) >= 7:
        gold_6m = _pct(gold, 6) or 0
        s["gold_6m"] = round(gold_6m * 100, 1)
        rate_lv      = s.get("rate_current") or 0
        # composite: gold up + rates elevated → inflationary
        raw = gold_6m * 3.0 + max(0.0, (rate_lv - 2.0) / 8.0)
        s["inflation_score"]  = max(-1.0, min(1.0, raw))
        s["inflation_signal"] = (
            "Elevated"  if raw >  0.25 else
            "Contained" if raw < -0.05 else
            "Moderate"
        )
    else:
        s.update(gold_6m=None, inflation_score=0.0, inflation_signal="—")

    # ── Dollar ────────────────────────────────────────────────────────────────
    dxy = macro_data.get("Dollar")
    if dxy is not None and len(dxy) >= 4:
        dxy_3m = _pct(dxy, 3) or 0
        s["dxy_3m"]       = round(dxy_3m * 100, 1)
        s["dxy_direction"] = (
            "Strengthening ↑" if dxy_3m >  0.01 else
            "Weakening ↓"     if dxy_3m < -0.01 else
            "Stable →"
        )
    else:
        s.update(dxy_3m=None, dxy_direction="—")

    # ── Oil ───────────────────────────────────────────────────────────────────
    oil = macro_data.get("Oil")
    s["oil_3m"] = round((_pct(oil, 3) or 0) * 100, 1) if oil is not None and len(oil) >= 4 else None

    # ── Dalio Cycle Phase ─────────────────────────────────────────────────────
    growth    = s.get("growth_signal",    "—")
    inflation = s.get("inflation_signal", "—")

    if growth == "Expanding" and inflation in ("Contained", "Moderate"):
        s.update(
            cycle_phase  = "Goldilocks",
            cycle_icon   = "🌤",
            cycle_color  = "#10b981",
            cycle_desc   = (
                "Strong growth with contained inflation — the ideal environment for risk assets. "
                "Equities, especially growth and high-beta names, tend to outperform."
            ),
            cycle_stocks = "Growth stocks, Technology, Consumer Discretionary",
            cycle_bonds  = "Bonds flat to negative. Equities preferred over fixed income.",
        )
    elif growth == "Expanding" and inflation == "Elevated":
        s.update(
            cycle_phase  = "Inflationary Boom",
            cycle_icon   = "🔥",
            cycle_color  = "#f59e0b",
            cycle_desc   = (
                "Growth with rising prices — central banks may tighten aggressively. "
                "Commodities, energy, and real assets outperform. Equity multiples compress."
            ),
            cycle_stocks = "Commodities, Energy, Materials, Financials (benefit from rising rates)",
            cycle_bonds  = "Nominal bonds underperform. Inflation-linked bonds (TIPS) and short-duration preferred.",
        )
    elif growth == "Contracting" and inflation == "Elevated":
        s.update(
            cycle_phase  = "Stagflation",
            cycle_icon   = "⚠️",
            cycle_color  = "#ef4444",
            cycle_desc   = (
                "The worst macro scenario: economic slowdown with persistent inflation. "
                "Very difficult for equities. Hard assets (gold, commodities) are relative safe havens."
            ),
            cycle_stocks = "Defensive sectors (Utilities, Healthcare, Consumer Staples), Gold, Commodities",
            cycle_bonds  = "Traditional bonds underperform. Short-duration and inflation-linked preferred.",
        )
    elif growth == "Contracting":
        s.update(
            cycle_phase  = "Deflationary Bust",
            cycle_icon   = "❄️",
            cycle_color  = "#6366f1",
            cycle_desc   = (
                "Recession risk or contraction with falling inflation. Flight to quality. "
                "Long-duration government bonds, gold, and defensive equities outperform."
            ),
            cycle_stocks = "Defensive sectors, Quality bonds, Gold — avoid cyclicals and high-beta names",
            cycle_bonds  = "Government bonds rally strongly. Long-duration bonds favoured.",
        )
    else:
        s.update(
            cycle_phase  = "Transition",
            cycle_icon   = "⚖️",
            cycle_color  = "#94a3b8",
            cycle_desc   = (
                "Mixed signals — the cycle phase is unclear. "
                "Maintain diversification and avoid large concentrated bets."
            ),
            cycle_stocks = "Balanced allocation. No strong directional conviction.",
            cycle_bonds  = "Balanced bond/equity mix.",
        )

    return s


def macro_implication(cycle_phase: str, beta: float, ticker: str) -> str:
    """
    Returns a one-paragraph plain-English implication of the macro cycle
    for the specific stock given its beta.
    """
    high_beta = beta > 1.2
    low_beta  = beta < 0.8

    if cycle_phase == "Goldilocks":
        if high_beta:
            return (
                f"**{ticker}** (β={beta:.2f}) is a high-beta stock in a Goldilocks environment — "
                "this is historically one of the best combinations. Risk appetite is high, growth is rewarded, "
                "and above-market-beta names tend to amplify the upside."
            )
        elif low_beta:
            return (
                f"**{ticker}** (β={beta:.2f}) is a defensive name in a Goldilocks environment. "
                "The stock should participate in the broad rally but may lag high-beta peers. "
                "Consider whether the defensive premium is justified at current valuations."
            )
        else:
            return (
                f"**{ticker}** (β={beta:.2f}) moves broadly with the market. "
                "In a Goldilocks environment, this is a solid position with balanced risk/reward."
            )
    elif cycle_phase == "Inflationary Boom":
        if high_beta:
            return (
                f"**{ticker}** (β={beta:.2f}) has elevated beta in an inflationary boom. "
                "Rising rates compress equity multiples — high-beta growth stocks face headwinds "
                "even if the economy continues to grow. Monitor rate sensitivity closely."
            )
        else:
            return (
                f"**{ticker}** (β={beta:.2f}) may hold up better than the broad market "
                "in an inflationary environment given its lower market sensitivity. "
                "Inflation pass-through capability and pricing power are the key differentiators."
            )
    elif cycle_phase == "Stagflation":
        return (
            f"**{ticker}** (β={beta:.2f}) faces a difficult macro backdrop. "
            "Stagflation is the toughest environment for equities regardless of beta. "
            "Focus on the company's pricing power, cash generation, and balance-sheet strength "
            "to assess resilience."
        )
    elif cycle_phase == "Deflationary Bust":
        if high_beta:
            return (
                f"**{ticker}** (β={beta:.2f}) is a high-beta name in a contractionary macro environment — "
                "historically the worst combination. The stock is likely to amplify broad-market downside. "
                "Consider position sizing and tail-risk hedging."
            )
        else:
            return (
                f"**{ticker}** (β={beta:.2f}) has a relatively low beta, which provides some cushion "
                "in a deflationary bust. The stock should outperform high-beta peers, though broad "
                "market declines will still create headwinds."
            )
    else:
        return (
            f"**{ticker}** (β={beta:.2f}): macro signals are mixed. "
            "The stock's performance will be driven more by company-specific factors than macro tailwinds. "
            "Focus on fundamentals and valuation."
        )
