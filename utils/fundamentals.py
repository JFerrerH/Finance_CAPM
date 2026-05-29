def parse_fundamentals(info):
    def _pct(v):
        return round(v * 100, 2) if v is not None else None

    def _fmt(v, decimals=2):
        return round(v, decimals) if v is not None else None

    valuation = {
        "P/E (TTM)":    _fmt(info.get("trailingPE")),
        "Forward P/E":  _fmt(info.get("forwardPE")),
        "P/B":          _fmt(info.get("priceToBook")),
        "P/S (TTM)":    _fmt(info.get("priceToSalesTrailing12Months")),
        "EV/EBITDA":    _fmt(info.get("enterpriseToEbitda")),
        "PEG Ratio":    _fmt(info.get("trailingPegRatio") or info.get("pegRatio")),
    }

    profitability = {
        "Gross Margin":     _pct(info.get("grossMargins")),
        "Operating Margin": _pct(info.get("operatingMargins")),
        "Net Margin":       _pct(info.get("profitMargins")),
        "ROE":              _pct(info.get("returnOnEquity")),
        "ROA":              _pct(info.get("returnOnAssets")),
    }

    leverage = {
        "Debt / Equity":  _fmt(info.get("debtToEquity")),
        "Current Ratio":  _fmt(info.get("currentRatio")),
        "Quick Ratio":    _fmt(info.get("quickRatio")),
        "Interest Coverage": _fmt(info.get("interestCoverage")),
    }

    growth = {
        "Revenue Growth (YoY)":  _pct(info.get("revenueGrowth")),
        "Earnings Growth (YoY)": _pct(info.get("earningsGrowth")),
        "EPS (TTM)":             _fmt(info.get("trailingEps")),
        "Forward EPS":           _fmt(info.get("forwardEps")),
    }

    return {
        "valuation":     valuation,
        "profitability": profitability,
        "leverage":      leverage,
        "growth":        growth,
    }


# Benchmark thresholds used for contextual help text on each metric
BENCHMARKS = {
    "P/E (TTM)":             "Sector-relative. Lower = cheaper vs earnings.",
    "Forward P/E":           "Market expectation. Compare to trailing P/E for growth premium.",
    "P/B":                   "< 1 may signal undervaluation. > 3 common in high-ROE firms.",
    "P/S (TTM)":             "< 2 generally conservative. High in growth sectors.",
    "EV/EBITDA":             "< 10 often considered fair; varies by industry.",
    "PEG Ratio":             "< 1 = potentially undervalued relative to growth. > 2 = pricey.",
    "Gross Margin":          "Higher is better. Technology > 60%, Manufacturing 20–40%.",
    "Operating Margin":      "> 15% healthy. Reflects operational efficiency.",
    "Net Margin":            "> 10% solid. Varies widely by sector.",
    "ROE":                   "> 15% strong. Buffett benchmark: consistent >20%.",
    "ROA":                   "> 5% good. Banks typically 1–2%.",
    "Debt / Equity":         "< 1 conservative. > 2 highly leveraged.",
    "Current Ratio":         "> 1.5 healthy. < 1 may indicate liquidity risk.",
    "Quick Ratio":           "> 1 healthy. Excludes inventory.",
    "Interest Coverage":     "> 3× safe. < 1.5× concerning.",
    "Revenue Growth (YoY)":  "Positive growth preferred. Context: sector growth rate.",
    "Earnings Growth (YoY)": "Positive preferred. Accelerating growth = bullish signal.",
    "EPS (TTM)":             "Earnings per share over last 12 months.",
    "Forward EPS":           "Analyst consensus EPS estimate for next 12 months.",
}
