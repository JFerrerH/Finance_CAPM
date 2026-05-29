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

def compute_cycle_signals(macro_data: dict, fred_data: dict | None = None) -> dict:
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

    # ── FRED Economic Data signals ────────────────────────────────────────────
    # Real economic data (CPI, TIPS breakevens, unemployment, industrial production)
    # that market prices alone cannot provide. When fred_data is absent the model
    # degrades gracefully to market-price-only signals.
    _fd = fred_data or {}
    s["fred_available"] = bool(_fd)

    # CPI: headline vs core divergence — the definitive supply-shock discriminator.
    # A wide headline-minus-core spread means inflation is in energy/food (supply side),
    # not in wages/rents/services (structural demand side).
    _cpi_hl   = _fd.get("CPI Headline")
    _cpi_core = _fd.get("CPI Core")
    cpi_hl_yoy   = round((_pct(_cpi_hl,   12) or 0) * 100, 2) if _cpi_hl   is not None and len(_cpi_hl)   >= 13 else None
    cpi_core_yoy = round((_pct(_cpi_core, 12) or 0) * 100, 2) if _cpi_core is not None and len(_cpi_core) >= 13 else None
    cpi_spread   = round(cpi_hl_yoy - cpi_core_yoy, 2) if (cpi_hl_yoy is not None and cpi_core_yoy is not None) else None
    s.update(cpi_headline_yoy=cpi_hl_yoy, cpi_core_yoy=cpi_core_yoy, cpi_spread=cpi_spread)

    # TIPS breakevens: market-implied inflation expectations.
    # If TIPS haven't repriced despite a commodity spike, bond markets don't believe
    # the inflation will persist — a strong transient/geopolitical signal.
    _tips5  = _fd.get("TIPS 5Y")
    _tips10 = _fd.get("TIPS 10Y")
    tips_5y       = round(float(_tips5.iloc[-1]),  2) if _tips5  is not None and not _tips5.empty  else None
    tips_10y      = round(float(_tips10.iloc[-1]), 2) if _tips10 is not None and not _tips10.empty else None
    tips_5y_3m    = round(_abs(_tips5,  3) or 0, 3)  if _tips5  is not None and len(_tips5)  >= 4  else None
    tips_10y_3m   = round(_abs(_tips10, 3) or 0, 3)  if _tips10 is not None and len(_tips10) >= 4  else None
    s.update(tips_5y=tips_5y, tips_10y=tips_10y, tips_5y_3m_chg=tips_5y_3m, tips_10y_3m_chg=tips_10y_3m)

    # Unemployment: rising unemployment contradicts wage-driven inflation
    # and confirms a demand-shock / labour-market-softening narrative.
    _unemp = _fd.get("Unemployment")
    unemp_current = round(float(_unemp.iloc[-1]), 1) if _unemp is not None and not _unemp.empty else None
    unemp_3m_chg  = round(_abs(_unemp, 3) or 0, 2)  if _unemp is not None and len(_unemp) >= 4  else None
    s.update(unemployment=unemp_current, unemployment_3m_chg=unemp_3m_chg)

    # Industrial Production: manufacturing activity proxy (PMI-like directional indicator).
    # Declining INDPRO during an apparent inflationary episode supports supply-shock diagnosis.
    _indpro = _fd.get("Industrial Prod")
    indpro_3m = round((_pct(_indpro, 3) or 0) * 100, 2) if _indpro is not None and len(_indpro) >= 4 else None
    s["indpro_3m"] = indpro_3m

    # ── Geopolitical supply-shock filter ──────────────────────────────────────
    # Combines market-price signals (Levels 1-3) with real economic data (Level 4)
    # to detect commodity surges driven by geopolitical events rather than structural
    # demand-pull inflation (e.g. Strait of Hormuz → oil spike → false Stagflation read).
    gold_3m_raw = _pct(gold, 3) if gold is not None and len(gold) >= 4 else None
    gold_3m_pct = (gold_3m_raw or 0.0) * 100        # e.g. 8.5 means +8.5%
    gold_6m_pct = s.get("gold_6m") or 0.0           # already in % pts
    oil_3m_pct  = s.get("oil_3m")  or 0.0           # already in % pts
    dxy_dir     = s.get("dxy_direction", "—")
    infl_sig    = s.get("inflation_signal", "—")

    s["gold_3m"] = round(gold_3m_pct, 1) if gold_3m_raw is not None else None

    # Level 1: Oil up >15% AND more than 2× gold's 3M move → supply shock fingerprint
    geo_oil_gold = oil_3m_pct > 15 and oil_3m_pct > max(gold_3m_pct, 1.0) * 2.0

    # Level 2: Dollar strengthening while inflation reads elevated → geopolitical risk-off,
    # not demand inflation (structural inflation is typically dollar-negative or mixed)
    geo_dxy_infl = infl_sig == "Elevated" and "Strengthening" in dxy_dir

    # Level 3: Is the inflation trend persistent (spread across 6M) or a recent spike (3M dominated)?
    # Persistent = the 3M contribution is ≤60% of the 6M move, meaning it built gradually
    inflation_persistent = (
        gold_6m_pct > 3
        and gold_3m_pct > 0
        and gold_3m_pct <= gold_6m_pct * 0.60
    )

    # Level 4: Real economic data confirmation (requires FRED)
    # CPI spread: headline running >1.5pp above core → energy/food driven, not wages/services
    geo_cpi_spread = (
        cpi_spread is not None
        and cpi_core_yoy is not None
        and cpi_spread > 1.5
        and cpi_core_yoy < 3.5
    )
    # TIPS stable: bond market not repricing inflation expectations despite commodity spike
    geo_tips_stable = (
        tips_5y is not None
        and tips_5y_3m is not None
        and abs(tips_5y_3m) < 0.30   # < 30 bp move in 3M
    )
    # If FRED confirms structural inflation (core entrenched + TIPS elevated), override persistence
    fred_confirms_structural = (
        cpi_core_yoy is not None and cpi_core_yoy > 4.0
        and tips_5y is not None and tips_5y > 3.0
    )
    if fred_confirms_structural:
        inflation_persistent = True   # real data overrides the market-price persistence filter

    geopolitical_warning   = geo_oil_gold or geo_dxy_infl or geo_cpi_spread
    s["inflation_signal_raw"] = infl_sig
    geo_adjustment_applied    = False

    # If the warning fires AND inflation is NOT confirmed structural → downgrade the signal
    if geopolitical_warning and not inflation_persistent and not fred_confirms_structural and infl_sig == "Elevated":
        s["inflation_signal"] = "Moderate"
        geo_adjustment_applied = True

    _geo_parts: list[str] = []
    if geo_oil_gold:
        _geo_parts.append(
            f"Oil +{oil_3m_pct:.0f}% vs Gold +{gold_3m_pct:.0f}% (3M) — "
            "commodity surge appears supply-driven, not demand-pull"
        )
    if geo_dxy_infl:
        _geo_parts.append(
            "Dollar strengthening while inflation reads elevated — "
            "consistent with geopolitical risk-off capital flight, not structural inflation"
        )
    if geo_cpi_spread and cpi_hl_yoy is not None and cpi_core_yoy is not None:
        _geo_parts.append(
            f"CPI headline {cpi_hl_yoy:.1f}% vs core {cpi_core_yoy:.1f}% "
            f"({cpi_spread:+.1f}pp spread) — inflation concentrated in energy/food, not broad-based"
        )
    if geo_tips_stable and geopolitical_warning and tips_5y is not None and tips_5y_3m is not None:
        _geo_parts.append(
            f"5Y TIPS breakeven at {tips_5y:.2f}% with only {tips_5y_3m * 100:+.0f}bp move in 3M — "
            "bond market not pricing persistent inflation"
        )
    if geopolitical_warning and not inflation_persistent and gold_6m_pct > 0 and not geo_cpi_spread:
        _geo_parts.append(
            f"Gold's recent 3M gain (+{gold_3m_pct:.0f}%) dominated its 6M trend "
            f"(+{gold_6m_pct:.0f}%) — suggests transient spike, not embedded inflation"
        )

    s["geopolitical_warning"]    = geopolitical_warning
    s["geo_oil_gold_signal"]     = geo_oil_gold
    s["geo_dxy_signal"]          = geo_dxy_infl
    s["geo_cpi_spread_signal"]   = geo_cpi_spread
    s["geo_tips_stable_signal"]  = geo_tips_stable
    s["fred_confirms_structural"] = fred_confirms_structural
    s["inflation_persistent"]    = inflation_persistent
    s["geo_adjustment_applied"]  = geo_adjustment_applied
    s["geo_shock_detail"]        = " · ".join(_geo_parts) if _geo_parts else ""

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

    # ── Cycle maturity assessment ─────────────────────────────────────────────
    s.update(_assess_maturity(s, macro_data))

    # ── Big Cycle (Dalio long-term debt cycle) ────────────────────────────────
    s.update(_compute_big_cycle(s, _fd))

    return s


# ─────────────────────────────────────────────────────────────────────────────
# Big Cycle — Dalio long-term debt supercycle positioning
# ─────────────────────────────────────────────────────────────────────────────

def _compute_big_cycle(s: dict, fred_data: dict) -> dict:
    """
    Estimates the current position in Dalio's Long-term Debt Cycle using
    federal debt/GDP, real interest rates, M2 money supply growth, and the
    Federal Funds Rate.

    The Big Cycle spans decades and describes the accumulation and eventual
    deleveraging of debt relative to income. It provides structural context
    for the short-term cycle — a Goldilocks reading at peak debt levels is
    far more fragile than the same reading in an early-expansion environment.

    Phases (Dalio, 'Principles for Navigating Big Debt Crises'):
        Early Expansion   — debt moderate and productive, few structural constraints
        Late Expansion    — debt elevated and still rising, fragility building
        Monetization/Top  — debt at extremes, central bank suppressing real rates
        At the Top        — peak leverage, tightening unusually painful
        Deleveraging      — debt/GDP declining through austerity / restructuring
        Beautiful Deleveraging — nominal growth > debt growth, soft landing

    Returns a flat dict prefixed with 'bc_' for easy identification.
    """
    bc: dict = {"bc_available": False}
    if not fred_data:
        return bc

    debt_gdp_s  = fred_data.get("Debt/GDP")
    m2_s        = fred_data.get("M2")
    fedfunds_s  = fred_data.get("Fed Funds")

    # Need at least debt/GDP to say anything meaningful
    if debt_gdp_s is None or debt_gdp_s.empty:
        return bc
    bc["bc_available"] = True

    # ── Debt / GDP ────────────────────────────────────────────────────────────
    debt_gdp_now = round(float(debt_gdp_s.iloc[-1]), 1)
    n_trend = min(36, len(debt_gdp_s) - 1)   # up to 3 years of trend
    debt_gdp_trend = round(float(debt_gdp_s.iloc[-1]) - float(debt_gdp_s.iloc[-1 - n_trend]), 1)
    bc["bc_debt_gdp"]       = debt_gdp_now
    bc["bc_debt_gdp_trend"] = debt_gdp_trend   # pp change over n_trend months

    # ── Real interest rate: nominal 10Y minus 10Y TIPS breakeven ─────────────
    rate_nominal = s.get("rate_current")          # already in % (e.g. 4.35)
    tips_10y_val = s.get("tips_10y")              # already in % from FRED
    if rate_nominal is not None and tips_10y_val is not None:
        real_rate = round(rate_nominal - tips_10y_val, 2)
    elif rate_nominal is not None:
        real_rate = round(rate_nominal - 2.5, 2)  # fallback: assume 2.5% expected inflation
    else:
        real_rate = None
    bc["bc_real_rate"] = real_rate

    # ── M2 growth (12-month % change) ─────────────────────────────────────────
    if m2_s is not None and len(m2_s) >= 13:
        bc["bc_m2_growth_12m"] = round((_pct(m2_s, 12) or 0) * 100, 1)
    else:
        bc["bc_m2_growth_12m"] = None

    # ── Federal Funds Rate ────────────────────────────────────────────────────
    if fedfunds_s is not None and not fedfunds_s.empty:
        bc["bc_fed_funds"] = round(float(fedfunds_s.iloc[-1]), 2)
    else:
        bc["bc_fed_funds"] = None

    # ── Phase classification ───────────────────────────────────────────────────
    debt_high      = debt_gdp_now > 90
    debt_very_high = debt_gdp_now > 115
    debt_rising    = debt_gdp_trend > 5
    debt_falling   = debt_gdp_trend < -5

    real_neg        = real_rate is not None and real_rate < 0
    real_deeply_neg = real_rate is not None and real_rate < -1.5
    real_positive   = real_rate is not None and real_rate > 0.5

    m2_growth  = bc.get("bc_m2_growth_12m") or 0
    m2_surging = m2_growth > 8

    if not debt_high and debt_rising:
        bc.update(
            bc_phase = "Early Expansion",
            bc_icon  = "🌱",
            bc_color = "#10b981",
            bc_desc  = (
                "Debt levels are moderate and rising productively. The long-term cycle "
                "is in early-to-mid expansion — credit is available, structural headwinds "
                "are limited, and short-cycle signals are reliable at face value."
            ),
            bc_short_cycle_note = "Short-cycle signals can be taken at face value — no structural debt overlay.",
        )
    elif debt_high and debt_rising and not debt_very_high:
        bc.update(
            bc_phase = "Late Expansion",
            bc_icon  = "⚡",
            bc_color = "#f59e0b",
            bc_desc  = (
                "Debt/GDP is elevated and still rising — hallmarks of late-cycle debt expansion. "
                "Short-term growth can persist, but structural fragility is building. "
                "Dalio identifies this as the period when debt service begins constraining future growth "
                "and central bank tightening becomes unusually potent."
            ),
            bc_short_cycle_note = "Apply a fragility discount to bullish short-cycle readings — debt levels limit policy response room.",
        )
    elif debt_very_high and (real_neg or m2_surging):
        bc.update(
            bc_phase = "Monetization / Top",
            bc_icon  = "🖨️",
            bc_color = "#ef4444",
            bc_desc  = (
                "Debt/GDP is at historically extreme levels and real interest rates are negative "
                "or money supply is expanding rapidly — the central bank is suppressing real rates "
                "to reduce the real burden of debt. This is Dalio's monetization phase: "
                "debt is being inflated away rather than paid down. "
                "Hard assets (gold, commodities, real estate) benefit; "
                "long-duration bonds face structural headwinds."
            ),
            bc_short_cycle_note = "Goldilocks readings in this context are structurally fragile — inflation is being used as a policy tool.",
        )
    elif debt_very_high and real_positive:
        bc.update(
            bc_phase = "At the Top",
            bc_icon  = "🔔",
            bc_color = "#ef4444",
            bc_desc  = (
                "Debt/GDP is at historically extreme levels with positive real interest rates — "
                "debt service costs are rising in real terms, which is unusually painful "
                "at these leverage levels. This is Dalio's 'at the top' configuration: "
                "further rate increases carry outsized systemic risk, "
                "and the next downturn may require aggressive policy intervention."
            ),
            bc_short_cycle_note = "High debt + positive real rates = tightening has maximum bite. Downturns may be amplified.",
        )
    elif debt_falling and real_deeply_neg and m2_surging:
        bc.update(
            bc_phase = "Beautiful Deleveraging",
            bc_icon  = "🔄",
            bc_color = "#6366f1",
            bc_desc  = (
                "Debt/GDP is declining while nominal growth (aided by money creation) exceeds "
                "debt growth — Dalio's 'beautiful deleveraging.' The pain of debt reduction "
                "is balanced by enough monetary stimulus to prevent deflationary collapse. "
                "This is the rare soft-landing path out of excessive debt."
            ),
            bc_short_cycle_note = "Regime is healing — short-cycle signals are gaining reliability as the debt overhang clears.",
        )
    elif debt_falling:
        bc.update(
            bc_phase = "Deleveraging",
            bc_icon  = "📉",
            bc_color = "#94a3b8",
            bc_desc  = (
                "Debt/GDP is declining — the system is in a deleveraging phase involving "
                "some combination of austerity, debt restructuring, wealth redistribution, "
                "and monetization. Growth is structurally subdued; short-cycle expansions "
                "tend to be shallower and more fragile than historical norms."
            ),
            bc_short_cycle_note = "Structural deleveraging suppresses the amplitude of short-cycle recoveries.",
        )
    else:
        bc.update(
            bc_phase = "Transition / Unclear",
            bc_icon  = "⚖️",
            bc_color = "#94a3b8",
            bc_desc  = "Big Cycle signals are mixed. Interpret short-cycle positioning without a strong long-term structural view.",
            bc_short_cycle_note = "",
        )

    return bc


# ─────────────────────────────────────────────────────────────────────────────
# Cycle maturity — where are we WITHIN the phase?
# ─────────────────────────────────────────────────────────────────────────────

def _assess_maturity(s: dict, macro_data: dict) -> dict:
    """
    Determines how far we are into the current cycle phase by comparing
    short-term (3M) vs medium-term (6M) momentum trajectories.

    Returns a dict with keys:
        cycle_stage         — "Early" | "Mid" | "Late" | "Unclear"
        cycle_stage_pct     — rough % through the phase (0-100)
        cycle_stage_color   — hex colour for the stage badge
        cycle_next_phase    — most likely next phase name
        cycle_next_icon     — icon for next phase
        cycle_watch_signal  — what to monitor for the transition
        cycle_stage_desc    — plain-English maturity narrative
        cycle_transition_risk — "Low" | "Moderate" | "High"
    """
    phase = s.get("cycle_phase", "Transition")

    # ── Short vs medium momentum for growth (S&P) ─────────────────────────────
    sp = macro_data.get("S&P 500")
    sp_3m     = _pct(sp, 3) or 0.0  if sp is not None and len(sp) >= 4  else 0.0
    sp_6m_val = _pct(sp, 6) or 0.0  if sp is not None and len(sp) >= 7  else 0.0
    # "accelerating" = last 3M contributed more than its proportional share
    growth_accel       = sp_3m > sp_6m_val / 2     if sp_6m_val > 0  else sp_3m > 0
    growth_decelerating = sp_3m < sp_6m_val / 3    if sp_6m_val > 0  else (sp_3m < 0 and sp_6m_val <= 0)
    growth_stabilising = sp_3m > sp_6m_val / 2     if sp_6m_val < 0  else False

    # ── Short vs medium momentum for inflation (Gold) ─────────────────────────
    gold      = macro_data.get("Gold")
    gold_3m   = _pct(gold, 3) or 0.0  if gold is not None and len(gold) >= 4  else 0.0
    gold_6m_v = _pct(gold, 6) or 0.0  if gold is not None and len(gold) >= 7  else 0.0
    infl_accel       = gold_3m > gold_6m_v / 2     if gold_6m_v > 0  else gold_3m > 0.02
    infl_decelerating = gold_3m < gold_6m_v / 3    if gold_6m_v > 0  else (gold_3m < 0 and gold_6m_v <= 0)

    # ── Rates ─────────────────────────────────────────────────────────────────
    rate_3m_bp  = s.get("rate_3m_bp") or 0
    rate_current = s.get("rate_current") or 0
    rate_rising  = rate_3m_bp > 15
    rate_falling = rate_3m_bp < -15
    rate_high    = rate_current > 4.5

    # ── VIX trend (proxy for fear building/subsiding) ─────────────────────────
    vix    = macro_data.get("VIX")
    vix_3m = _pct(vix, 3) or 0.0  if vix is not None and len(vix) >= 4  else 0.0
    vix_rising  = vix_3m >  0.15   # VIX up >15% in 3M = stress building
    vix_falling = vix_3m < -0.15   # VIX down >15% = fear subsiding

    # ── FRED-derived maturity signals (None-safe: all False when FRED unavailable) ──
    # These supplement the market-price signals above without breaking behaviour
    # when no FRED key is configured.
    unrate_rising   = (s.get("unemployment_3m_chg") or 0) >  0.30  # labour market softening
    unrate_falling  = (s.get("unemployment_3m_chg") or 0) < -0.30  # labour market recovering
    indpro_weak     = (s.get("indpro_3m") or 0) < -0.50            # industrial activity declining
    indpro_strong   = (s.get("indpro_3m") or 0) >  0.50            # industrial activity recovering
    cpi_core_high   = (s.get("cpi_core_yoy") or 0) >  4.0          # core CPI entrenched → more tightening
    tips_5y_low     = 0 < (s.get("tips_5y") or 0) < 2.50           # inflation expectations anchored
    tips_5y_high    = (s.get("tips_5y") or 0) > 3.0                # inflation expectations elevated

    # ── Phase-specific maturity scoring ──────────────────────────────────────
    m: dict = {}

    if phase == "Goldilocks":
        # Late signs: rates rising, gold picking up (inflation building), VIX bottoming,
        # labour softening (unrate rising = demand starting to crack under rate pressure)
        late_count = sum([rate_rising, gold_3m > 0.05, vix_rising, infl_accel, unrate_rising])
        # Early signs: growth re-accelerating, fear subsiding, rates not yet rising,
        # inflation expectations still anchored (TIPS low = no structural inflation fear yet)
        early_count = sum([growth_accel, vix_falling, not rate_rising, tips_5y_low])

        if late_count >= 2:
            m.update(
                cycle_stage        = "Late",
                cycle_stage_pct    = 78,
                cycle_stage_color  = "#f97316",
                cycle_next_phase   = "Inflationary Boom",
                cycle_next_icon    = "🔥",
                cycle_watch_signal = "Gold/inflation trend + 10Y yield trajectory — if both keep rising, transition is imminent",
                cycle_stage_desc   = (
                    "We appear to be in **late Goldilocks**. Rates are rising and/or inflation indicators "
                    "are creeping up, signalling the sweet spot is narrowing. "
                    "The window for high-beta, growth-oriented positioning may be closing."
                ),
                cycle_transition_risk = "High — transition likely within 1–3 months",
            )
        elif early_count >= 2:
            m.update(
                cycle_stage        = "Early",
                cycle_stage_pct    = 22,
                cycle_stage_color  = "#34d399",
                cycle_next_phase   = "Inflationary Boom",
                cycle_next_icon    = "🔥",
                cycle_watch_signal = "Watch for inflation indicators (gold, commodity prices) starting to build",
                cycle_stage_desc   = (
                    "We appear to be in **early Goldilocks**. Growth is re-accelerating, "
                    "inflation is benign, and risk appetite is recovering. "
                    "This is historically the strongest entry point for growth equities."
                ),
                cycle_transition_risk = "Low — phase likely has room to run",
            )
        else:
            m.update(
                cycle_stage        = "Mid",
                cycle_stage_pct    = 50,
                cycle_stage_color  = "#10b981",
                cycle_next_phase   = "Inflationary Boom",
                cycle_next_icon    = "🔥",
                cycle_watch_signal = "Monitor 10Y yield direction and gold trend for early signs of inflation build-up",
                cycle_stage_desc   = (
                    "We appear to be in **mid-Goldilocks** — the sweet spot. "
                    "Growth is solid, inflation contained, and risk assets are supported. "
                    "Conditions remain favourable but watch for overheating signals."
                ),
                cycle_transition_risk = "Moderate — watch closely",
            )

    elif phase == "Inflationary Boom":
        # Late signs: growth rolling over, rates very high, labour market softening,
        # core CPI entrenched (means Fed has no off-ramp — tightening must continue)
        growth_fading = sp_3m < 0 and sp_6m_val > -0.03   # recent months flipping negative
        late_count  = sum([rate_high and rate_rising, growth_fading or growth_decelerating,
                           vix_rising, cpi_core_high, unrate_rising])
        # Early signs: growth still strong, rates not yet biting, industrial activity healthy
        early_count = sum([growth_accel, not rate_high, infl_accel and not growth_fading,
                           indpro_strong])

        if late_count >= 2:
            next_p = "Stagflation" if not infl_decelerating else "Deflationary Bust"
            m.update(
                cycle_stage        = "Late",
                cycle_stage_pct    = 80,
                cycle_stage_color  = "#ef4444",
                cycle_next_phase   = next_p,
                cycle_next_icon    = "⚠️" if next_p == "Stagflation" else "❄️",
                cycle_watch_signal = "Watch for GDP/PMI data deteriorating while CPI stays elevated — that is the stagflation trigger",
                cycle_stage_desc   = (
                    "We appear to be in **late Inflationary Boom**. Growth is losing momentum "
                    "while inflation stays elevated and rates are biting. "
                    f"**{next_p} risk is rising.** Consider reducing high-beta exposure and rotating to value/defensive."
                ),
                cycle_transition_risk = "High — transition may be imminent",
            )
        elif early_count >= 2:
            m.update(
                cycle_stage        = "Early",
                cycle_stage_pct    = 25,
                cycle_stage_color  = "#f59e0b",
                cycle_next_phase   = "Stagflation",
                cycle_next_icon    = "⚠️",
                cycle_watch_signal = "Watch for rate hikes starting to slow credit growth and consumer spending",
                cycle_stage_desc   = (
                    "We appear to be in **early Inflationary Boom**. Growth is still strong "
                    "and inflation just broke higher. Commodities and value are outperforming; "
                    "rate-sensitive growth stocks are beginning to lag."
                ),
                cycle_transition_risk = "Low-Moderate — still early innings",
            )
        else:
            m.update(
                cycle_stage        = "Mid",
                cycle_stage_pct    = 52,
                cycle_stage_color  = "#f59e0b",
                cycle_next_phase   = "Stagflation",
                cycle_next_icon    = "⚠️",
                cycle_watch_signal = "Monitor economic growth data — any sustained slowdown signals the next transition",
                cycle_stage_desc   = (
                    "We appear to be in **mid-Inflationary Boom**. Growth holds but inflation "
                    "is the dominant theme. Central banks are likely in active tightening mode."
                ),
                cycle_transition_risk = "Moderate — depends on central bank response",
            )

    elif phase == "Stagflation":
        # Late signs: inflation rolling over (gold falling, core CPI peaking), growth stabilising
        # TIPS elevated then falling = inflation expectations finally cracking → pivot approaching
        inflation_rolling_over = infl_decelerating or gold_3m < -0.02
        late_count  = sum([inflation_rolling_over, growth_stabilising, vix_falling,
                           tips_5y_high and infl_decelerating])
        # Early signs: inflation still accelerating, growth decelerating, industrial activity weak
        early_count = sum([not inflation_rolling_over, growth_decelerating, vix_rising, indpro_weak])

        if late_count >= 2:
            m.update(
                cycle_stage        = "Late",
                cycle_stage_pct    = 80,
                cycle_stage_color  = "#a78bfa",
                cycle_next_phase   = "Deflationary Bust",
                cycle_next_icon    = "❄️",
                cycle_watch_signal = "Watch inflation rolling over sharply — that is the signal for the rate-cut pivot",
                cycle_stage_desc   = (
                    "We appear to be in **late Stagflation**. Inflation may be peaking and growth "
                    "has already deteriorated. The macro is setting up for a policy pivot — "
                    "once inflation falls enough for central banks to cut, the next phase begins."
                ),
                cycle_transition_risk = "High — watch for central bank pivot",
            )
        elif early_count >= 2:
            m.update(
                cycle_stage        = "Early",
                cycle_stage_pct    = 20,
                cycle_stage_color  = "#ef4444",
                cycle_next_phase   = "Deflationary Bust",
                cycle_next_icon    = "❄️",
                cycle_watch_signal = "Monitor whether central banks choose to prioritise fighting inflation over growth",
                cycle_stage_desc   = (
                    "We appear to be in **early Stagflation**. The growth slowdown is just beginning "
                    "while inflation remains high. Asset preservation and hard assets are the priority."
                ),
                cycle_transition_risk = "High — worst phase for most assets",
            )
        else:
            m.update(
                cycle_stage        = "Mid",
                cycle_stage_pct    = 55,
                cycle_stage_color  = "#ef4444",
                cycle_next_phase   = "Deflationary Bust",
                cycle_next_icon    = "❄️",
                cycle_watch_signal = "Track peak inflation data closely — the exact CPI peak defines the exit",
                cycle_stage_desc   = (
                    "We appear to be in **mid-Stagflation**. Both growth and asset prices are under pressure. "
                    "Central banks face the hardest trade-off: fight inflation or support growth."
                ),
                cycle_transition_risk = "High",
            )

    elif phase == "Deflationary Bust":
        # Late signs: growth stabilising, VIX falling, rates down, industrial activity recovering,
        # unemployment falling (labour market turning — the clearest leading recovery signal)
        late_count  = sum([growth_stabilising or growth_accel, vix_falling, rate_falling,
                           indpro_strong, unrate_falling])
        # Early signs: growth declining, fear building, industrial activity still weak
        early_count = sum([growth_decelerating, vix_rising, not rate_falling, indpro_weak])

        if late_count >= 2:
            m.update(
                cycle_stage        = "Late",
                cycle_stage_pct    = 82,
                cycle_stage_color  = "#34d399",
                cycle_next_phase   = "Goldilocks",
                cycle_next_icon    = "🌤",
                cycle_watch_signal = "Watch for S&P stabilisation + credit spreads narrowing — those confirm the turn",
                cycle_stage_desc   = (
                    "We appear to be in **late Deflationary Bust**. The worst may be behind us — "
                    "rates are falling, fear is subsiding, and growth is stabilising. "
                    "Early positioning in quality growth names may be rewarded."
                ),
                cycle_transition_risk = "Moderate — recovery forming, but false dawns are common",
            )
        elif early_count >= 2:
            m.update(
                cycle_stage        = "Early",
                cycle_stage_pct    = 18,
                cycle_stage_color  = "#6366f1",
                cycle_next_phase   = "Goldilocks",
                cycle_next_icon    = "🌤",
                cycle_watch_signal = "Watch credit markets and leading indicators for the first signs of stabilisation",
                cycle_stage_desc   = (
                    "We appear to be in **early Deflationary Bust**. Economic contraction is beginning, "
                    "equities are retreating, and fear is rising. Capital preservation is paramount."
                ),
                cycle_transition_risk = "High — downturn deepening",
            )
        else:
            m.update(
                cycle_stage        = "Mid",
                cycle_stage_pct    = 55,
                cycle_stage_color  = "#6366f1",
                cycle_next_phase   = "Goldilocks",
                cycle_next_icon    = "🌤",
                cycle_watch_signal = "Monetary stimulus works with a 6–12 month lag — watch leading indicators",
                cycle_stage_desc   = (
                    "We appear to be in **mid-Deflationary Bust**. Downturn deepening, "
                    "central banks cutting aggressively. Defensive assets and quality bonds outperform."
                ),
                cycle_transition_risk = "Moderate",
            )

    else:  # Transition / Flat
        m.update(
            cycle_stage        = "Unclear",
            cycle_stage_pct    = 50,
            cycle_stage_color  = "#94a3b8",
            cycle_next_phase   = "Undetermined",
            cycle_next_icon    = "⚖️",
            cycle_watch_signal = "Track S&P 6M momentum (growth) and gold/rate level (inflation) for clearer signals",
            cycle_stage_desc   = (
                "Mixed signals — the cycle phase is unclear. "
                "Monitor trend direction over the next 4–6 weeks before taking a directional view."
            ),
            cycle_transition_risk = "Moderate",
        )

    return m


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
