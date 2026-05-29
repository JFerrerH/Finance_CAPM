"""
utils/scoring.py
----------------
Computes the five-dimension investment scorecard that drives the
📍 Thesis page. Each dimension returns (score: int 1-5, detail: str).

Dimensions:
    1. Alpha Quality   — Jensen's alpha vs 0
    2. Risk / Return   — Sharpe + max drawdown
    3. Momentum        — Annualised return vs market
    4. Financial Health — Revenue growth, margins, FCF, leverage
    5. Valuation       — Upside/downside to blended fair value
"""

from utils.financials import calc_cagr

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(x: float, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, int(round(x))))


SCORE_COLORS = {
    5: "#10b981",   # green
    4: "#34d399",   # light green
    3: "#f59e0b",   # amber
    2: "#f97316",   # orange
    1: "#ef4444",   # red
}

SCORE_LABELS = {
    5: "Excellent",
    4: "Good",
    3: "Neutral",
    2: "Weak",
    1: "Poor",
}

VERDICT_MAP = [
    (4.5, "Strong Buy",  "#10b981"),
    (3.5, "Buy",         "#34d399"),
    (2.5, "Hold",        "#f59e0b"),
    (1.5, "Sell",        "#f97316"),
    (0.0, "Strong Sell", "#ef4444"),
]

# Dimension weights: de-weight Alpha (noisy) and Momentum (overlaps Alpha),
# up-weight Financials (fundamentals) and Risk/Return (sustained performance).
# Macro Fit is market-wide context, not company-specific — lowest weight.
SCORE_WEIGHTS = {
    "Alpha Quality":    0.15,
    "Risk / Return":    0.20,
    "Momentum":         0.10,
    "Financial Health": 0.25,
    "Valuation":        0.20,
    "Macro Fit":        0.10,
}

# Sector tilt per Dalio cycle phase: +1 if sector benefits, -1 if sector suffers.
# Applied on top of the beta-based score to capture structural industry dynamics.
SECTOR_CYCLE_TILT = {
    "Goldilocks": {
        "positive": {"Technology", "Consumer Cyclical", "Communication Services",
                     "Financial Services", "Financials", "Industrials"},
        "negative": {"Utilities", "Consumer Defensive"},
    },
    "Inflationary Boom": {
        "positive": {"Energy", "Materials", "Real Estate",
                     "Financial Services", "Financials"},
        "negative": {"Technology", "Consumer Cyclical", "Communication Services"},
    },
    "Stagflation": {
        "positive": {"Energy", "Materials", "Consumer Defensive",
                     "Healthcare", "Utilities"},
        "negative": {"Technology", "Real Estate", "Consumer Cyclical",
                     "Communication Services", "Financial Services", "Financials"},
    },
    "Deflationary Bust": {
        "positive": {"Consumer Defensive", "Healthcare", "Utilities"},
        "negative": {"Energy", "Materials", "Financial Services", "Financials",
                     "Real Estate", "Consumer Cyclical", "Industrials"},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Individual dimension scorers
# ─────────────────────────────────────────────────────────────────────────────

def score_alpha(capm: dict) -> tuple[int, str]:
    """
    Jensen's alpha quality, penalised for statistical insignificance.
    A large but noisy alpha (high p-value) is capped — it may just be luck.
    """
    alpha = capm.get("jensen_alpha_annual") or 0.0
    pval  = capm.get("alpha_pvalue")

    # Confidence cap: limits maximum score when alpha is statistically weak
    if pval is None or pval < 0.05:
        cap = 5
        sig = f", p={pval:.2f} ✓" if pval is not None else ""
    elif pval < 0.10:
        cap = 4
        sig = f", p={pval:.2f} marginal"
    else:
        cap = 3
        sig = f", p={pval:.2f} — not significant"

    if   alpha >  0.05: raw, label = 5, f"Strong alpha ({alpha:+.1%}/yr{sig})"
    elif alpha >  0.02: raw, label = 4, f"Positive alpha ({alpha:+.1%}/yr{sig})"
    elif alpha >  0.00: raw, label = 3, f"Marginal alpha ({alpha:+.1%}/yr{sig})"
    elif alpha > -0.02: raw, label = 2, f"Slight drag ({alpha:+.1%}/yr{sig})"
    else:               raw, label = 1, f"Negative alpha ({alpha:+.1%}/yr{sig})"

    return min(raw, cap), label


def score_risk_return(perf: dict) -> tuple[int, str]:
    """Sharpe ratio + maximum drawdown composite."""
    sharpe = perf.get("sharpe") or 0.0
    mdd    = abs(perf.get("max_drawdown") or 1.0)
    detail = f"Sharpe {sharpe:.2f} | Max DD {mdd:.0%}"
    if   sharpe > 1.5 and mdd < 0.20: return 5, detail
    elif sharpe > 1.0 and mdd < 0.30: return 4, detail
    elif sharpe > 0.5:                 return 3, detail
    elif sharpe > 0.0:                 return 2, detail
    else:                              return 1, detail


def score_momentum(perf: dict, capm: dict) -> tuple[int, str]:
    """Annualised return excess vs market return."""
    ann_ret = perf.get("ann_return") or 0.0
    mkt_ret = capm.get("Rm") or 0.0
    excess  = ann_ret - mkt_ret
    if   excess >  0.10: return 5, f"Beating market by {excess:+.1%}"
    elif excess >  0.05: return 4, f"Beating market by {excess:+.1%}"
    elif excess >  0.00: return 3, f"Slight outperformance ({excess:+.1%})"
    elif excess > -0.05: return 2, f"Lagging market by {abs(excess):.1%}"
    else:                return 1, f"Significantly lagging ({excess:+.1%})"


def score_financial_health(
    inc: dict, cf: dict | None, bal: dict | None
) -> tuple[int | None, str]:
    """Revenue growth + earnings growth + margin + FCF + leverage."""
    if not inc or not inc.get("revenue"):
        return None, "Financial data unavailable"

    pts = 3.0
    detail_parts: list[str] = []

    # Revenue CAGR
    rev_cagr = calc_cagr(inc.get("revenue"))
    if rev_cagr is not None:
        detail_parts.append(f"Rev CAGR {rev_cagr:+.0%}")
        if   rev_cagr >  0.12: pts += 0.8
        elif rev_cagr >  0.05: pts += 0.3
        elif rev_cagr <  0.00: pts -= 1.0

    # Net income CAGR
    ni_cagr = calc_cagr(inc.get("net_income"))
    if ni_cagr is not None:
        if   ni_cagr >  0.12: pts += 0.5
        elif ni_cagr <  0.00: pts -= 0.5

    # Net margin (most recent year)
    rev_list = inc.get("revenue") or []
    ni_list  = inc.get("net_income") or []
    if rev_list and ni_list and rev_list[-1]:
        margin = ni_list[-1] / rev_list[-1]
        detail_parts.append(f"Margin {margin:.0%}")
        if   margin >  0.20: pts += 0.5
        elif margin >  0.10: pts += 0.2
        elif margin <  0.00: pts -= 0.8

    # FCF generation
    fcf_list = (cf or {}).get("fcf") or []
    if fcf_list and fcf_list[-1] is not None:
        if   fcf_list[-1] > 0: pts += 0.3; detail_parts.append("FCF ✓")
        else:                  pts -= 0.5; detail_parts.append("FCF ✗")

    # D/E ratio
    debt_list   = (bal or {}).get("total_debt") or []
    equity_list = (bal or {}).get("equity")     or []
    if debt_list and equity_list and equity_list[-1] and equity_list[-1] != 0:
        de = debt_list[-1] / equity_list[-1]
        detail_parts.append(f"D/E {de:.1f}x")
        if   de > 3.0: pts -= 0.5
        elif de < 1.0: pts += 0.2

    return _clamp(pts), " | ".join(detail_parts[:4]) or "Limited data"


def score_valuation(
    current_price: float | None,
    dcf_price: float | None,
    pe_fair_price: float | None,
    ddm_price: float | None = None,
) -> tuple[int | None, str]:
    """
    Upside / downside to blended fair value estimate.
    DDM is included when available (most relevant for dividend-paying companies).
    """
    estimates = [p for p in [dcf_price, pe_fair_price, ddm_price] if p and p > 0]
    if not estimates or not current_price:
        return None, "Valuation data unavailable"

    avg_fair  = sum(estimates) / len(estimates)
    upside    = (avg_fair - current_price) / current_price
    methods   = sum([bool(dcf_price), bool(pe_fair_price), bool(ddm_price and ddm_price > 0)])
    blend_str = f"{methods}-method blend" if methods > 1 else "single method"

    if   upside >  0.30: return 5, f"~{upside:.0%} upside ({blend_str})"
    elif upside >  0.10: return 4, f"~{upside:.0%} upside ({blend_str})"
    elif upside > -0.10: return 3, f"Near fair value ({upside:+.0%}, {blend_str})"
    elif upside > -0.25: return 2, f"~{abs(upside):.0%} overvalued ({blend_str})"
    else:                return 1, f"~{abs(upside):.0%} overvalued — significant downside"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def score_macro_fit(signals: dict, beta: float, sector: str = "") -> tuple[int, str]:
    """
    Scores how well the stock fits the current macro cycle using both beta
    and sector (sector has a structural relationship with each cycle phase
    that beta alone cannot capture, e.g. Energy in Inflationary Boom).
    """
    phase     = signals.get("cycle_phase", "Transition")
    rate_dir  = signals.get("rate_direction", "Stable →")
    icon      = signals.get("cycle_icon", "")
    rising    = "Rising" in rate_dir
    high_beta = beta > 1.3
    low_beta  = beta < 0.8

    # Base score from beta × cycle phase
    if phase == "Goldilocks":
        if high_beta:  base, note = 5, "high-beta — historically optimal"
        elif low_beta: base, note = 3, "defensive beta may lag the rally"
        else:          base, note = 4, "favourable for equities"

    elif phase == "Inflationary Boom":
        if rising and high_beta:
            base, note = 2, "rising rates compress high-beta multiples"
        elif low_beta:
            base, note = 4, "defensive names hold up better"
        else:
            base, note = 3, "mixed — watch rate sensitivity"

    elif phase == "Stagflation":
        if high_beta:  base, note = 1, "high-beta — worst combination"
        elif low_beta: base, note = 3, "defensives provide relative shelter"
        else:          base, note = 2, "difficult for most equities"

    elif phase == "Deflationary Bust":
        if high_beta:  base, note = 1, "high-beta amplifies downside"
        elif low_beta: base, note = 4, "defensive beta limits losses"
        else:          base, note = 2, "broad equity headwinds"

    else:
        return 3, f"{icon} Cycle unclear — macro neither tailwind nor headwind"

    # Sector tilt: structural industry dynamics on top of beta signal
    tilt       = SECTOR_CYCLE_TILT.get(phase, {})
    sector_adj = 0
    sector_note = ""
    if sector in tilt.get("positive", set()):
        sector_adj  =  1
        sector_note = f"; {sector} structurally favoured in {phase}"
    elif sector in tilt.get("negative", set()):
        sector_adj  = -1
        sector_note = f"; {sector} structurally challenged in {phase}"

    final = _clamp(base + sector_adj)
    return final, f"{icon} {phase} — {note}{sector_note}"


def get_verdict(scores: dict) -> tuple[str, str, float]:
    """
    Weighted-average score → (verdict label, colour, weighted_avg).

    Weights from SCORE_WEIGHTS de-emphasise statistically noisy dimensions
    (Alpha, Momentum) and up-weight fundamentals (Financial Health, Valuation).

    Floor rule: a company with severe fundamental deterioration (Financial Health = 1
    AND Valuation = 1) cannot be rated Buy or above regardless of macro/momentum —
    those signals cannot compensate for a structurally broken business.
    """
    weighted_sum  = 0.0
    total_weight  = 0.0
    for key, val in scores.items():
        if val is not None and key in SCORE_WEIGHTS:
            weighted_sum += val * SCORE_WEIGHTS[key]
            total_weight += SCORE_WEIGHTS[key]

    if total_weight == 0:
        return "Insufficient Data", "#94a3b8", 0.0

    avg = weighted_sum / total_weight   # normalised so missing dimensions don't distort

    # Floor: structurally broken fundamentals cap the verdict at Hold
    fin  = scores.get("Financial Health")
    val  = scores.get("Valuation")
    if fin is not None and val is not None and fin <= 1 and val <= 1 and avg > 2.49:
        avg = 2.49

    for threshold, label, color in VERDICT_MAP:
        if avg >= threshold:
            return label, color, round(avg, 2)
    return "Strong Sell", "#ef4444", round(avg, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Auto bull / bear case generator
# ─────────────────────────────────────────────────────────────────────────────

def build_bull_bear(
    capm: dict,
    perf: dict,
    var_results: dict,
    inc: dict,
    cf: dict | None,
    bal: dict | None,
    valuation_upside: float | None,
    cycle_phase: str,
    beta: float,
) -> tuple[list[str], list[str]]:
    """
    Returns (bull_points, bear_points) — each a list of plain strings.
    """
    bull: list[str] = []
    bear: list[str] = []

    # ── Alpha ──────────────────────────────────────────────────────────────────
    alpha = capm.get("jensen_alpha_annual") or 0.0
    if alpha > 0.03:
        bull.append(f"Historically generates {alpha:+.1%}/yr risk-adjusted alpha vs the market")
    elif alpha < -0.03:
        bear.append(f"Negative historical alpha of {alpha:+.1%}/yr — destroys risk-adjusted value")

    # ── Sharpe & drawdown ──────────────────────────────────────────────────────
    sharpe = perf.get("sharpe") or 0.0
    mdd    = perf.get("max_drawdown") or 0.0
    if sharpe > 1.2:
        bull.append(f"Strong Sharpe ratio of {sharpe:.2f} — well-compensated for risk taken")
    if abs(mdd) > 0.40:
        bear.append(f"History of severe drawdowns (max loss: {mdd:.0%}) — high volatility risk")

    # ── VaR ───────────────────────────────────────────────────────────────────
    var95 = var_results.get("var_95") or 0.0
    if var95 < -0.10:
        bear.append(f"Monthly 95% VaR of {var95:.1%} — significant tail-risk exposure")

    # ── Beta ──────────────────────────────────────────────────────────────────
    if beta > 1.5:
        bear.append(f"High market beta ({beta:.2f}) amplifies broad-market downturns")
    elif beta < 0.7:
        bull.append(f"Defensive beta ({beta:.2f}) — cushions against market sell-offs")

    # ── Revenue growth ────────────────────────────────────────────────────────
    rev_cagr = calc_cagr((inc or {}).get("revenue"))
    if rev_cagr and rev_cagr > 0.10:
        bull.append(f"Revenue compounding at {rev_cagr:.0%} CAGR — strong topline momentum")
    elif rev_cagr and rev_cagr < 0:
        bear.append(f"Declining revenue ({rev_cagr:.0%} CAGR) — topline under pressure")

    # ── Margin ────────────────────────────────────────────────────────────────
    rev_list = (inc or {}).get("revenue") or []
    ni_list  = (inc or {}).get("net_income") or []
    if rev_list and ni_list and rev_list[-1]:
        margin = ni_list[-1] / rev_list[-1]
        if margin > 0.20:
            bull.append(f"High-quality business with {margin:.0%} net margin")
        elif margin < 0:
            bear.append(f"Company is currently loss-making ({margin:.0%} net margin)")

    # ── FCF ───────────────────────────────────────────────────────────────────
    fcf_list = (cf or {}).get("fcf") or []
    if fcf_list and fcf_list[-1]:
        if fcf_list[-1] > 0:
            bull.append("Positive free cash flow — self-funding growth without dilution")
        else:
            bear.append("Negative free cash flow — relies on external financing")

    # ── Leverage ──────────────────────────────────────────────────────────────
    debt_list   = (bal or {}).get("total_debt") or []
    equity_list = (bal or {}).get("equity")     or []
    if debt_list and equity_list and equity_list[-1] and equity_list[-1] != 0:
        de = debt_list[-1] / equity_list[-1]
        if de > 3.0:
            bear.append(f"High leverage (D/E {de:.1f}x) raises refinancing risk in a rising-rate environment")
        elif de < 0.5:
            bull.append(f"Clean balance sheet (D/E {de:.1f}x) — financial resilience and optionality")

    # ── Valuation ─────────────────────────────────────────────────────────────
    if valuation_upside is not None:
        if valuation_upside > 0.20:
            bull.append(f"Appears ~{valuation_upside:.0%} undervalued vs blended fair value estimate")
        elif valuation_upside < -0.20:
            bear.append(f"Appears ~{abs(valuation_upside):.0%} overvalued — limited margin of safety")

    # ── Macro ──────────────────────────────────────────────────────────────────
    if cycle_phase == "Goldilocks" and beta > 1.0:
        bull.append(f"Macro tailwind: Goldilocks environment favours high-beta names like this one")
    elif cycle_phase in ("Stagflation", "Deflationary Bust") and beta > 1.2:
        bear.append(f"Macro headwind: {cycle_phase} environment is adverse for high-beta equities")
    elif cycle_phase == "Goldilocks" and beta < 0.8:
        bear.append("Defensive nature may underperform in a risk-on Goldilocks regime")

    return bull[:5], bear[:5]
